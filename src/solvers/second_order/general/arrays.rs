use ndarray::{
    Array, ArrayD, ArrayView, ArrayViewD, ArrayViewMut, ArrayViewMut1, ArrayViewMutD, Dimension,
    IxDyn,
};

use super::{SecondOrderOdeProblem, SecondOrderSolution};
use crate::solvers::second_order::function::{ArrayAcceleration, SecondOrderFunction};
use crate::{
    CallbackAction, CallbackSave, ConfigurationError, EventCrossing, EventDirection,
    InterpolationError, SolveError,
};

impl SecondOrderOdeProblem<(), ()> {
    /// Creates a problem with ndarray scalar, vector, or matrix partitions.
    ///
    /// The function overwrites acceleration and receives velocity before
    /// position. Both initial arrays must have exactly the same shape. The
    /// adapters preserve logical indices, including nonstandard input layouts,
    /// without allocating during acceleration evaluation for fixed-rank arrays.
    ///
    /// ```
    /// use differential_equations::ndarray::{arr0, ArrayView0, ArrayViewMut0};
    /// use differential_equations::solvers::second_order::{
    ///     SecondOrderOdeProblem, VelocityVerlet, solve_second_order,
    /// };
    /// use differential_equations::SolveOptions;
    /// let problem = SecondOrderOdeProblem::from_array(
    ///     |mut a: ArrayViewMut0<'_, f64>, _: ArrayView0<'_, f64>, q: ArrayView0<'_, f64>, _: &(), _| {
    ///         a[[]] = -q[[]];
    ///     },
    ///     arr0(0.0), arr0(1.0), (0.0, 1.0), (),
    /// )?;
    /// let options = SolveOptions::new().with_adaptive(false).with_initial_step(0.01);
    /// let solution = solve_second_order(&problem, VelocityVerlet, &options)?;
    /// assert!((solution.last_position_array()[[]] - 1.0_f64.cos()).abs() < 1e-4);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_array<F, P, D>(
        acceleration: F,
        initial_velocity: Array<f64, D>,
        initial_position: Array<f64, D>,
        time_span: (f64, f64),
        parameters: P,
    ) -> Result<SecondOrderOdeProblem<impl SecondOrderFunction<P>, P>, ConfigurationError>
    where
        D: Dimension,
        F: for<'a, 'b, 'c> Fn(
            ArrayViewMut<'a, f64, D>,
            ArrayView<'b, f64, D>,
            ArrayView<'c, f64, D>,
            &P,
            f64,
        ),
    {
        from_array_function(
            move |output: ArrayViewMut<'_, f64, D>,
                  velocity: ArrayView<'_, f64, D>,
                  position: ArrayView<'_, f64, D>,
                  parameters: &P,
                  time| {
                acceleration(output, velocity, position, parameters, time);
                Ok(())
            },
            initial_velocity,
            initial_position,
            time_span,
            parameters,
        )
    }

    /// Creates a problem whose function returns an ndarray acceleration.
    ///
    /// Velocity and position must have exactly the same shape. A returned
    /// acceleration with any other shape produces
    /// [`SolveError::DerivativeShapeMismatch`], including during structural
    /// Jacobian or steady-state callback evaluations. Returning an owned array
    /// may allocate each evaluation; [`Self::from_array`] is the in-place form.
    pub fn from_array_out_of_place<F, P, D>(
        acceleration: F,
        initial_velocity: Array<f64, D>,
        initial_position: Array<f64, D>,
        time_span: (f64, f64),
        parameters: P,
    ) -> Result<SecondOrderOdeProblem<impl SecondOrderFunction<P>, P>, ConfigurationError>
    where
        D: Dimension,
        F: for<'a, 'b> Fn(ArrayView<'a, f64, D>, ArrayView<'b, f64, D>, &P, f64) -> Array<f64, D>,
    {
        from_array_function(
            move |mut output: ArrayViewMut<'_, f64, D>,
                  velocity: ArrayView<'_, f64, D>,
                  position: ArrayView<'_, f64, D>,
                  parameters: &P,
                  time| {
                let result = acceleration(velocity, position, parameters, time);
                if result.raw_dim() != output.raw_dim() {
                    return Err(SolveError::DerivativeShapeMismatch);
                }
                output.assign(&result);
                Ok(())
            },
            initial_velocity,
            initial_position,
            time_span,
            parameters,
        )
    }
}

fn from_array_function<F, P, D: Dimension>(
    function: F,
    initial_velocity: Array<f64, D>,
    initial_position: Array<f64, D>,
    time_span: (f64, f64),
    parameters: P,
) -> Result<SecondOrderOdeProblem<ArrayAcceleration<F, D>, P>, ConfigurationError> {
    let shape = initial_position.raw_dim();
    if initial_velocity.raw_dim() != shape {
        return Err(ConfigurationError::DimensionMismatch {
            context: "second-order state partitions",
        });
    }
    let mut problem = SecondOrderOdeProblem::new(
        ArrayAcceleration {
            function,
            shape: shape.clone(),
        },
        initial_velocity.iter().copied().collect::<Vec<_>>(),
        initial_position.iter().copied().collect::<Vec<_>>(),
        time_span,
        parameters,
    );
    problem.state_shape = shape.into_dyn();
    Ok(problem)
}

fn view<'a>(shape: &IxDyn, values: &'a [f64]) -> ArrayViewD<'a, f64> {
    ArrayViewD::from_shape(shape.clone(), values)
        .expect("partition shape must match its validated storage")
}

fn view_mut<'a>(shape: &IxDyn, values: &'a mut [f64]) -> ArrayViewMutD<'a, f64> {
    ArrayViewMutD::from_shape(shape.clone(), values)
        .expect("partition shape must match its validated storage")
}

fn condition<F, P, R>(shape: IxDyn, function: F) -> impl Fn(&[f64], &[f64], &P, f64) -> R + 'static
where
    F: for<'a, 'b> Fn(ArrayViewD<'a, f64>, ArrayViewD<'b, f64>, &P, f64) -> R + 'static,
{
    move |velocity, position, parameters, time| {
        function(
            view(&shape, velocity),
            view(&shape, position),
            parameters,
            time,
        )
    }
}

fn affect<F, P>(
    shape: IxDyn,
    function: F,
) -> impl Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static
where
    F: for<'a, 'b> Fn(ArrayViewMutD<'a, f64>, ArrayViewMutD<'b, f64>, &P, f64) -> CallbackAction
        + 'static,
{
    move |velocity, position, parameters, time| {
        function(
            view_mut(&shape, velocity),
            view_mut(&shape, position),
            parameters,
            time,
        )
    }
}

impl<F, P> SecondOrderOdeProblem<F, P> {
    /// Shape of each partition; an empty slice denotes an ndarray scalar.
    pub fn state_shape(&self) -> &[usize] {
        self.state_shape.slice()
    }

    /// Initial velocity as a shape-preserving ndarray view.
    pub fn initial_velocity_array(&self) -> ArrayViewD<'_, f64> {
        // The flat constructor allows mismatched partitions until solve-time
        // validation. Its velocity view must still be safe to inspect.
        if self.initial_velocity.len() != self.initial_position.len() {
            return ArrayView::from(self.initial_velocity.as_slice()).into_dyn();
        }
        view(&self.state_shape, &self.initial_velocity)
    }

    /// Initial position as a shape-preserving ndarray view.
    pub fn initial_position_array(&self) -> ArrayViewD<'_, f64> {
        view(&self.state_shape, &self.initial_position)
    }

    /// Adds a shape-aware discrete callback, with velocity before position.
    pub fn with_array_discrete_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: for<'a, 'b> Fn(ArrayViewD<'a, f64>, ArrayViewD<'b, f64>, &P, f64) -> bool + 'static,
        A: for<'a, 'b> Fn(
                ArrayViewMutD<'a, f64>,
                ArrayViewMutD<'b, f64>,
                &P,
                f64,
            ) -> CallbackAction
            + 'static,
    {
        self.with_array_discrete_callback_saving(CallbackSave::After, condition, affect)
    }

    /// Adds a shape-aware discrete callback with explicit saving behavior.
    pub fn with_array_discrete_callback_saving<C, A>(
        self,
        save: CallbackSave,
        test: C,
        effect: A,
    ) -> Self
    where
        C: for<'a, 'b> Fn(ArrayViewD<'a, f64>, ArrayViewD<'b, f64>, &P, f64) -> bool + 'static,
        A: for<'a, 'b> Fn(
                ArrayViewMutD<'a, f64>,
                ArrayViewMutD<'b, f64>,
                &P,
                f64,
            ) -> CallbackAction
            + 'static,
    {
        let shape = self.state_shape.clone();
        self.with_discrete_callback_saving(
            save,
            condition(shape.clone(), test),
            affect(shape, effect),
        )
    }

    /// Adds a shape-aware callback at exact preset times.
    pub fn with_array_preset_time_callback<A>(
        self,
        times: impl IntoIterator<Item = f64>,
        affect: A,
    ) -> Self
    where
        A: for<'a, 'b> Fn(
                ArrayViewMutD<'a, f64>,
                ArrayViewMutD<'b, f64>,
                &P,
                f64,
            ) -> CallbackAction
            + 'static,
    {
        self.with_array_preset_time_callback_saving(times, CallbackSave::After, affect)
    }

    /// Adds a shape-aware preset-time callback with explicit saving behavior.
    pub fn with_array_preset_time_callback_saving<A>(
        self,
        times: impl IntoIterator<Item = f64>,
        save: CallbackSave,
        effect: A,
    ) -> Self
    where
        A: for<'a, 'b> Fn(
                ArrayViewMutD<'a, f64>,
                ArrayViewMutD<'b, f64>,
                &P,
                f64,
            ) -> CallbackAction
            + 'static,
    {
        let shape = self.state_shape.clone();
        self.with_preset_time_callback_saving(times, save, affect(shape, effect))
    }

    /// Adds a shape-aware zero-crossing callback in either direction.
    pub fn with_array_continuous_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: for<'a, 'b> Fn(ArrayViewD<'a, f64>, ArrayViewD<'b, f64>, &P, f64) -> f64 + 'static,
        A: for<'a, 'b> Fn(
                ArrayViewMutD<'a, f64>,
                ArrayViewMutD<'b, f64>,
                &P,
                f64,
            ) -> CallbackAction
            + 'static,
    {
        self.with_array_continuous_callback_saving(CallbackSave::Both, condition, affect)
    }

    /// Adds a shape-aware zero-crossing callback with explicit saving behavior.
    pub fn with_array_continuous_callback_saving<C, A>(
        self,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: for<'a, 'b> Fn(ArrayViewD<'a, f64>, ArrayViewD<'b, f64>, &P, f64) -> f64 + 'static,
        A: for<'a, 'b> Fn(
                ArrayViewMutD<'a, f64>,
                ArrayViewMutD<'b, f64>,
                &P,
                f64,
            ) -> CallbackAction
            + 'static,
    {
        self.with_array_continuous_callback_direction_saving(
            EventDirection::Any,
            save,
            condition,
            affect,
        )
    }

    /// Adds a direction-filtered shape-aware zero-crossing callback.
    pub fn with_array_continuous_callback_direction<C, A>(
        self,
        direction: EventDirection,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: for<'a, 'b> Fn(ArrayViewD<'a, f64>, ArrayViewD<'b, f64>, &P, f64) -> f64 + 'static,
        A: for<'a, 'b> Fn(
                ArrayViewMutD<'a, f64>,
                ArrayViewMutD<'b, f64>,
                &P,
                f64,
            ) -> CallbackAction
            + 'static,
    {
        self.with_array_continuous_callback_direction_saving(
            direction,
            CallbackSave::Both,
            condition,
            affect,
        )
    }

    /// Adds a direction-filtered shape-aware callback with explicit saving behavior.
    pub fn with_array_continuous_callback_direction_saving<C, A>(
        self,
        direction: EventDirection,
        save: CallbackSave,
        test: C,
        effect: A,
    ) -> Self
    where
        C: for<'a, 'b> Fn(ArrayViewD<'a, f64>, ArrayViewD<'b, f64>, &P, f64) -> f64 + 'static,
        A: for<'a, 'b> Fn(
                ArrayViewMutD<'a, f64>,
                ArrayViewMutD<'b, f64>,
                &P,
                f64,
            ) -> CallbackAction
            + 'static,
    {
        let shape = self.state_shape.clone();
        self.with_continuous_callback_direction_saving(
            direction,
            save,
            condition(shape.clone(), test),
            affect(shape, effect),
        )
    }

    /// Adds several zero-crossing conditions over shape-aware partitions.
    pub fn with_array_vector_continuous_callback<C, A>(
        self,
        event_count: usize,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: for<'a, 'b, 'c> Fn(
                ArrayViewMut1<'a, f64>,
                ArrayViewD<'b, f64>,
                ArrayViewD<'c, f64>,
                &P,
                f64,
            ) + 'static,
        A: for<'a, 'b> Fn(
                ArrayViewMutD<'a, f64>,
                ArrayViewMutD<'b, f64>,
                &P,
                f64,
                &[EventCrossing],
            ) -> CallbackAction
            + 'static,
    {
        self.with_array_vector_continuous_callback_saving(
            event_count,
            CallbackSave::Both,
            condition,
            affect,
        )
    }

    /// Adds shape-aware vector conditions with explicit saving behavior.
    pub fn with_array_vector_continuous_callback_saving<C, A>(
        self,
        event_count: usize,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: for<'a, 'b, 'c> Fn(
                ArrayViewMut1<'a, f64>,
                ArrayViewD<'b, f64>,
                ArrayViewD<'c, f64>,
                &P,
                f64,
            ) + 'static,
        A: for<'a, 'b> Fn(
                ArrayViewMutD<'a, f64>,
                ArrayViewMutD<'b, f64>,
                &P,
                f64,
                &[EventCrossing],
            ) -> CallbackAction
            + 'static,
    {
        let condition_shape = self.state_shape.clone();
        let affect_shape = self.state_shape.clone();
        self.with_vector_continuous_callback_saving(
            event_count,
            save,
            move |output, velocity, position, parameters, time| {
                condition(
                    ArrayViewMut1::from(output),
                    view(&condition_shape, velocity),
                    view(&condition_shape, position),
                    parameters,
                    time,
                )
            },
            move |velocity, position, parameters, time, events| {
                affect(
                    view_mut(&affect_shape, velocity),
                    view_mut(&affect_shape, position),
                    parameters,
                    time,
                    events,
                )
            },
        )
    }
}

impl SecondOrderSolution {
    /// Shape of each partition; an empty slice denotes an ndarray scalar.
    pub fn state_shape(&self) -> &[usize] {
        self.state_shape.slice()
    }

    /// Saved velocity as a shape-preserving ndarray view.
    pub fn velocity_array(&self, index: usize) -> Option<ArrayViewD<'_, f64>> {
        self.velocity(index)
            .map(|values| view(&self.state_shape, values))
    }

    /// Saved position as a shape-preserving ndarray view.
    pub fn position_array(&self, index: usize) -> Option<ArrayViewD<'_, f64>> {
        self.position(index)
            .map(|values| view(&self.state_shape, values))
    }

    /// Last saved velocity as a shape-preserving ndarray view.
    pub fn last_velocity_array(&self) -> ArrayViewD<'_, f64> {
        view(&self.state_shape, self.last_velocity())
    }

    /// Last saved position as a shape-preserving ndarray view.
    pub fn last_position_array(&self) -> ArrayViewD<'_, f64> {
        view(&self.state_shape, self.last_position())
    }

    /// Interpolates shape-preserving `(velocity, position)` arrays.
    pub fn interpolate_array(
        &self,
        time: f64,
    ) -> Result<(ArrayD<f64>, ArrayD<f64>), InterpolationError> {
        let (velocity, position) = self.try_interpolate(time)?;
        let reshape = |values| {
            ArrayD::from_shape_vec(self.state_shape.clone(), values)
                .map_err(|_| InterpolationError::DimensionMismatch)
        };
        Ok((reshape(velocity)?, reshape(position)?))
    }
}
