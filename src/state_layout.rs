//! Explicit matrix-to-state layout adapters.
//!
//! ndarray's standard layout is row-major; nalgebra and numeris matrix slices
//! are column-major. A raw slice alone does not identify its logical layout.
//! These borrowed views preserve that distinction without an allocation.

use std::{error::Error, fmt};

/// Logical order used when flattening or restoring a matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixOrder {
    /// Index `(row, col)` is `row * columns + col`.
    RowMajor,
    /// Index `(row, col)` is `col * rows + row`.
    ColumnMajor,
}

/// Invalid matrix extent, stride, or output buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LayoutError {
    /// Shape or stride arithmetic overflowed `usize`.
    Overflow,
    /// Storage is too short for the requested shape and strides.
    StorageTooShort {
        /// Minimum backing storage length.
        required: usize,
        /// Supplied backing storage length.
        actual: usize,
    },
    /// A flatten/unflatten output has the wrong length.
    LengthMismatch {
        /// Required transfer length.
        expected: usize,
        /// Supplied transfer length.
        actual: usize,
    },
    /// A logical index lies outside the view.
    IndexOutOfBounds {
        /// Requested zero-based row.
        row: usize,
        /// Requested zero-based column.
        column: usize,
    },
}
impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => write!(f, "matrix shape or strides overflow"),
            Self::StorageTooShort { required, actual } => {
                write!(f, "matrix storage needs {required} elements, got {actual}")
            }
            Self::LengthMismatch { expected, actual } => {
                write!(f, "matrix transfer needs {expected} elements, got {actual}")
            }
            Self::IndexOutOfBounds { row, column } => {
                write!(f, "matrix index ({row}, {column}) is outside the view")
            }
        }
    }
}
impl Error for LayoutError {}

/// Borrowed logical matrix with explicit nonnegative element strides.
///
/// Empty shapes are valid. Padding and submatrix strides are allowed; repeated
/// elements (zero or overlapping strides) are also permitted for read-only views.
#[derive(Clone, Copy, Debug)]
pub struct MatrixView<'a> {
    data: &'a [f64],
    rows: usize,
    columns: usize,
    row_stride: usize,
    column_stride: usize,
}
impl<'a> MatrixView<'a> {
    /// Views contiguous row-major or column-major data without copying.
    pub fn new(
        data: &'a [f64],
        rows: usize,
        columns: usize,
        order: MatrixOrder,
    ) -> Result<Self, LayoutError> {
        let (rs, cs) = match order {
            MatrixOrder::RowMajor => (columns, 1),
            MatrixOrder::ColumnMajor => (1, rows),
        };
        Self::strided(data, rows, columns, rs, cs)
    }
    /// Views a possibly padded or transposed matrix; strides count elements.
    pub fn strided(
        data: &'a [f64],
        rows: usize,
        columns: usize,
        row_stride: usize,
        column_stride: usize,
    ) -> Result<Self, LayoutError> {
        rows.checked_mul(columns).ok_or(LayoutError::Overflow)?;
        let required = if rows == 0 || columns == 0 {
            0
        } else {
            (rows - 1)
                .checked_mul(row_stride)
                .and_then(|r| {
                    (columns - 1)
                        .checked_mul(column_stride)
                        .and_then(|c| r.checked_add(c))
                })
                .and_then(|n| n.checked_add(1))
                .ok_or(LayoutError::Overflow)?
        };
        if data.len() < required {
            return Err(LayoutError::StorageTooShort {
                required,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            rows,
            columns,
            row_stride,
            column_stride,
        })
    }
    /// Logical dimensions.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
    /// Returns one element, with checked logical indices.
    pub fn get(&self, row: usize, column: usize) -> Option<f64> {
        (row < self.rows && column < self.columns)
            .then(|| self.data[row * self.row_stride + column * self.column_stride])
    }
    /// Transposes the logical view without moving any data.
    pub fn transpose(self) -> Self {
        Self {
            data: self.data,
            rows: self.columns,
            columns: self.rows,
            row_stride: self.column_stride,
            column_stride: self.row_stride,
        }
    }
    /// Copies into an exactly sized reusable slice in the requested order.
    /// Validation occurs before writing, and this operation never allocates.
    pub fn flatten_into(&self, output: &mut [f64], order: MatrixOrder) -> Result<(), LayoutError> {
        let expected = self.rows * self.columns;
        if output.len() != expected {
            return Err(LayoutError::LengthMismatch {
                expected,
                actual: output.len(),
            });
        }
        for row in 0..self.rows {
            for column in 0..self.columns {
                let index = match order {
                    MatrixOrder::RowMajor => row * self.columns + column,
                    MatrixOrder::ColumnMajor => column * self.rows + row,
                };
                output[index] = self.data[row * self.row_stride + column * self.column_stride];
            }
        }
        Ok(())
    }
    /// Copies into a fixed array, checking its compile-time length at runtime.
    pub fn flatten_array<const N: usize>(
        &self,
        order: MatrixOrder,
    ) -> Result<[f64; N], LayoutError> {
        let mut result = [0.0; N];
        self.flatten_into(&mut result, order)?;
        Ok(result)
    }
    /// Restores a borrowed flat state into a fixed row-major rectangular array.
    pub fn to_rows<const R: usize, const C: usize>(&self) -> Result<[[f64; C]; R], LayoutError> {
        if self.rows != R || self.columns != C {
            return Err(LayoutError::LengthMismatch {
                expected: R.checked_mul(C).ok_or(LayoutError::Overflow)?,
                actual: self.rows * self.columns,
            });
        }
        let mut rows = [[0.0; C]; R];
        for (r, row) in rows.iter_mut().enumerate() {
            for (c, value) in row.iter_mut().enumerate() {
                *value = self.data[r * self.row_stride + c * self.column_stride];
            }
        }
        Ok(rows)
    }
}

/// Converts between explicit contiguous layouts using caller-owned buffers.
///
/// This is both flatten and unflatten: choose the source and destination logical
/// orders explicitly. No third-party matrix package or allocation is required.
pub fn convert_matrix_layout(
    source: &[f64],
    destination: &mut [f64],
    rows: usize,
    columns: usize,
    source_order: MatrixOrder,
    destination_order: MatrixOrder,
) -> Result<(), LayoutError> {
    MatrixView::new(source, rows, columns, source_order)?
        .flatten_into(destination, destination_order)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangular_nonsymmetric_round_trip() {
        let row = [1., 2., 3., 4., 5., 6.];
        let view = MatrixView::new(&row, 2, 3, MatrixOrder::RowMajor).unwrap();
        let column = view.flatten_array::<6>(MatrixOrder::ColumnMajor).unwrap();
        assert_eq!(column, [1., 4., 2., 5., 3., 6.]);
        let restored = MatrixView::new(&column, 2, 3, MatrixOrder::ColumnMajor)
            .unwrap()
            .to_rows::<2, 3>()
            .unwrap();
        assert_eq!(restored, [[1., 2., 3.], [4., 5., 6.]]);
        assert_eq!(
            view.transpose()
                .flatten_array::<6>(MatrixOrder::RowMajor)
                .unwrap(),
            column
        );
    }
    #[test]
    fn padded_views_and_reusable_buffers() {
        let data = [1., 2., 99., 3., 4., 99., 5., 6.];
        let view = MatrixView::strided(&data, 3, 2, 3, 1).unwrap();
        let mut out = [0.; 6];
        view.flatten_into(&mut out, MatrixOrder::ColumnMajor)
            .unwrap();
        assert_eq!(out, [1., 3., 5., 2., 4., 6.]);
        let mut short = [9.; 5];
        assert!(
            view.flatten_into(&mut short, MatrixOrder::RowMajor)
                .is_err()
        );
        assert_eq!(short, [9.; 5]);
        assert_eq!(view.get(3, 0), None);
    }
    #[test]
    fn shapes_are_checked_before_access() {
        assert!(MatrixView::new(&[], usize::MAX, 2, MatrixOrder::RowMajor).is_err());
        assert!(MatrixView::strided(&[], 2, 2, usize::MAX, 1).is_err());
        assert!(MatrixView::new(&[0.; 5], 2, 3, MatrixOrder::RowMajor).is_err());
        assert_eq!(
            MatrixView::new(&[], 0, 3, MatrixOrder::RowMajor)
                .unwrap()
                .flatten_array::<0>(MatrixOrder::RowMajor)
                .unwrap(),
            [0.0_f64; 0]
        );
    }
}
