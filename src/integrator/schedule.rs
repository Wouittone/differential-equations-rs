use crate::callback::CallbackOutcome;

/// Tracks directionally ordered integration times that a driver must hit.
///
/// Specialized drivers share this small scheduler with the common first-order
/// driver so `SolveOptions::time_stops` has identical clipping semantics
/// without copying stop-search logic into every integration loop.
pub(crate) struct TimeStopSchedule<'a> {
    stops: &'a [f64],
    next: usize,
    end: f64,
    direction: f64,
}

pub(crate) fn callback_adjusted_step(
    callbacks: CallbackOutcome,
    proposed_step: f64,
    direction: f64,
    maximum_step: f64,
) -> f64 {
    let mut magnitude = callbacks
        .requested_step
        .unwrap_or_else(|| proposed_step.abs())
        .min(maximum_step);
    if let Some(limit) = callbacks.step_limit {
        magnitude = magnitude.min(limit);
    }
    direction * magnitude
}

impl<'a> TimeStopSchedule<'a> {
    pub(crate) fn new(stops: &'a [f64], start: f64, end: f64) -> Self {
        let direction = (end - start).signum();
        let mut schedule = Self {
            stops,
            next: 0,
            end,
            direction,
        };
        schedule.accepted(start);
        schedule
    }

    pub(crate) fn clip_step_with(&self, time: f64, step: f64, additional_stop: Option<f64>) -> f64 {
        let scheduled = self.stops.get(self.next).copied().unwrap_or(self.end);
        let target = additional_stop.map_or(scheduled, |additional| {
            if self.direction * (additional - scheduled) < 0.0 {
                additional
            } else {
                scheduled
            }
        });
        if self.direction * (time + step - target) > 0.0 {
            target - time
        } else {
            step
        }
    }

    pub(crate) fn accepted(&mut self, time: f64) {
        while self
            .stops
            .get(self.next)
            .is_some_and(|stop| self.direction * (*stop - time) <= 0.0)
        {
            self.next += 1;
        }
    }
}
