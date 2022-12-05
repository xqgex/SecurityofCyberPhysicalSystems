""" A simple implementation of a square matrix.

This code was proudly written using Gedit.
Released under the following license:
https://github.com/xqgex/SecurityofCyberPhysicalSystems/blob/main/LICENSE
"""

from math import sqrt as math_sqrt
from typing import List, Tuple

import attr

from .data_types import Matrix, Vector

_INVERSE_ROUND_PRECISION = 14


@attr.s(frozen=True, slots=True)
class SquareMatrix(Matrix):
    """ Implementation of a square matrix. """

    def __attrs_post_init__(self) -> None:
        row_lengths = tuple(map(len, self))
        if len(row_lengths) == 0:
            raise ValueError('A square matrix of size 0 is not allowed.')
        if len(row_lengths) != row_lengths[0]:
            raise ValueError(
                'A square matrix should have the same number of rows and columns, '
                f'got an {row_lengths[0]}x{len(row_lengths)} matrix\n{str(self)}')

    def expand_with_indices(self, indices: Tuple[int, ...], size: int) -> 'Matrix':
        """
        >>> print(SquareMatrix.from_lists(((1., 2.), (3., 4.))).expand_with_indices((2, 4), 6))
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 3.0, 0.0, 4.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        """
        def _choose_value(row_index: int, column_index: int) -> bool:
            return row_index in indices and column_index in indices
        if len(indices) != len(self):
            raise ValueError(f'Cannot expand square matrix with indices of different size.\n{self}\n{indices}')
        self_iterator = (c for r in self for c in r)
        return self.evolve(matrix=tuple(Vector.from_list(tuple(next(self_iterator) if _choose_value(r_i, c_i) else 0.0
                                                         for c_i in range(size)))
                                        for r_i in range(size)))

    def inverse(self) -> 'SquareMatrix':
        """ Calculate the inverse matrix. """
        return SquareMatrix.from_lists(inverse_gauss_jordan(tuple(map(tuple, self))))

    def square_root(self) -> 'SquareMatrix':
        r""" Return the square root for a Matrix instance.

        >>> print(SquareMatrix.from_lists([[4., 8.], [8., 80.]]).square_root())
        [[2.0, 0.0],
         [4.0, 8.0]]
        >>> print(SquareMatrix.from_lists([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]]).square_root())
        [[2.0, 0.0, 0.0],
         [6.0, 1.0, 0.0],
         [-8.0, 5.0, 3.0]]
        >>> l = [[1., 2., 4., 7.], [2., 13., 23., 38.], [4., 23., 77., 122.], [7., 38., 122., 294.]]
        >>> print(SquareMatrix.from_lists(l).square_root())
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 3.0, 0.0, 0.0],
         [4.0, 5.0, 6.0, 0.0],
         [7.0, 8.0, 9.0, 10.0]]

        .. math::
           LL^{T} = A \Rightarrow \sqrt{A} = L
        """
        if self.is_zero_matrix():
            return SquareMatrix.of_size(len(self))  # The square root of the zero matrix is a zero matrix
        if any(self[i][i] == 0.0 for i in range(len(self))):
            raise ValueError(f'Cannot calculate square root for square matrix with zero(s) on the diagonal.\n{self}')
        return SquareMatrix.from_lists(square_root_cholesky_banachiewicz(tuple(map(tuple, self))))


def inverse_gauss_jordan(left_matrix: Tuple[Tuple[float, ...], ...]) -> Tuple[Tuple[float, ...], ...]:
    """ Inverse of a matrix using elementary row operations (Gauss-Jordan method).

    >>> inverse_gauss_jordan(((4., 7.), (2., 6.)))
    ((0.6, -0.7), (-0.2, 0.4))
    >>> inverse_gauss_jordan(((3., 3.5), (3.2, 3.6)))
    ((-9.0, 8.75), (8.0, -7.5))
    """
    matrix_dimension = len(left_matrix)
    right_matrix = tuple(tuple(1. if i == j else 0. for i in range(matrix_dimension)) for j in range(matrix_dimension))
    for index_1 in range(matrix_dimension):
        if left_matrix[index_1][index_1] == 0.:
            left_matrix, right_matrix = _switch_rows(left_matrix, right_matrix, index_1)
        for index_2 in range(index_1 + 1, matrix_dimension):
            left_matrix, right_matrix = _eliminate_row(left_matrix, right_matrix, index_2, index_1, 0.)
    for index_1 in reversed(range(1, matrix_dimension)):
        for index_2 in reversed(range(index_1)):
            left_matrix, right_matrix = _eliminate_row(left_matrix, right_matrix, index_2, index_1, 0.)
    for index in range(matrix_dimension):
        left_matrix, right_matrix = _eliminate_row(left_matrix, right_matrix, index, index, 1.)
    return _round_matrix_values(right_matrix)


def square_root_cholesky_banachiewicz(matrix_a: Tuple[Tuple[float, ...], ...]) -> List[List[float]]:
    r""" The Cholesky–Banachiewicz algorithm starts from the upper left corner of the matrix L and proceeds to calculate
         the matrix row by row.

    :complexity: :math:`O(\frac{1}{3} n^3 + \frac{2}{3} n^2)`
    :equation:

    .. math::
       L_{i,j} = \begin{cases}
         \sqrt{A_{j,j} - \sum_{k=1}^{j - 1} L_{j,k}^2} & j = i \\
         \frac{1}{L_{j,j}} (A_{i,j} - \sum_{k=1}^{j - 1} L_{i,k} L_{j,k}) & j < i
       \end{cases}
    """
    def _raise_no_square_root() -> None:
        raise ValueError(f'Cannot find the square root of the given matrix\nA matrix: {matrix_a}\nL matrix: {matrix_l}')
    matrix_l = [[0.0 for _ in range(len(matrix_a))] for _ in range(len(matrix_a))]
    for i in range(len(matrix_a)):  # pylint: disable=consider-using-enumerate
        for j in range(i + 1):
            sum1 = sum(matrix_l[i][k] * matrix_l[j][k] for k in range(j))
            if i == j:
                if matrix_a[j][j] < sum1:
                    _raise_no_square_root()
                matrix_l[i][j] = math_sqrt(matrix_a[j][j] - sum1)
            else:
                if matrix_l[j][j] == 0.0:
                    _raise_no_square_root()
                matrix_l[i][j] = 1.0 / matrix_l[j][j] * (matrix_a[i][j] - sum1)
    return matrix_l


def _eliminate_row(
        left_matrix: Tuple[Tuple[float, ...], ...],
        right_matrix: Tuple[Tuple[float, ...], ...],
        target_row: int,
        target_column: int,
        target_value: float
        ) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[Tuple[float, ...], ...]]:
    """ Use `left_matrix[target_column]` to set `target_column` at the `left_matrix[target_row]` to be `target_value`.

    >>> _eliminate_row(((4., 7.), (2., 6.)), ((1., 0.), (0., 1.)), 1, 0, 0.)
    (((4.0, 7.0), (0.0, 2.5)), ((1.0, 0.0), (-0.5, 1.0)))
    >>> _eliminate_row(((4., 7.), (2., 6.)), ((1., 0.), (0., 1.)), 1, 0, 1.)
    (((4.0, 7.0), (1.0, 4.25)), ((1.0, 0.0), (-0.25, 1.0)))
    >>> _eliminate_row(((3, 3.5), (3.2, 3.6)), ((1., 0.), (0, 1)), 1, 0, 0.)
    (((3, 3.5), (0.0, -0.1333333333333333)), ((1.0, 0.0), (-1.0666666666666667, 1.0)))
    """
    def _calculate_matrix(matrix: Tuple[Tuple[float, ...], ...], factor: float) -> Tuple[Tuple[float, ...], ...]:
        return tuple(tuple(c - (factor * matrix[target_column][c_i]) if r_i == target_row else c
                           for c_i, c in enumerate(r))
                     for r_i, r in enumerate(matrix))
    factor = (left_matrix[target_row][target_column] - target_value) / left_matrix[target_column][target_column]
    return _calculate_matrix(left_matrix, factor),  _calculate_matrix(right_matrix, factor)


def _round_matrix_values(matrix: Tuple[Tuple[float, ...], ...]) -> Tuple[Tuple[float, ...], ...]:
    """
    >>> _round_matrix_values(((0.6000000000000001, -0.7000000000000002), (-9.000000000000004, 8.750000000000004)))
    ((0.6, -0.7), (-9.0, 8.75))
    >>> _round_matrix_values(((0.9999999999999998, 4.44089209850062e-16), (-2.220446049250313e-16, 1.0000000000000004)))
    ((1.0, 0.0), (-0.0, 1.0))
    """
    return tuple(tuple(round(c, _INVERSE_ROUND_PRECISION) for c in r) for r in matrix)


def _switch_rows(
        left_matrix: Tuple[Tuple[float, ...], ...],
        right_matrix: Tuple[Tuple[float, ...], ...],
        row_number: int
        ) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[Tuple[float, ...], ...]]:
    """
    >>> _switch_rows(((0., 3.5), (3.2, 3.6)), ((1., 0.), (0., 1.)), 0)
    (((3.2, 3.6), (0.0, 3.5)), ((0.0, 1.0), (1.0, 0.0)))
    """
    def _switch(
            matrix: Tuple[Tuple[float, ...], ...],
            row_1_index: int,
            row_2_index: int
            ) -> Tuple[Tuple[float, ...], ...]:
        return tuple(matrix[row_2_index] if i == row_1_index else matrix[row_1_index] if i == row_2_index else r
                     for i, r in enumerate(matrix))
    for index in range(row_number + 1, len(left_matrix)):
        if left_matrix[row_number][index] != 0.:
            return _switch(left_matrix, row_number, index), _switch(right_matrix, row_number, index)
    raise ValueError('Matrix is not invertible')
