""" Python implementation for Matrix and Vector classes.

This code was proudly written using Gedit.
Released under the following license:
https://github.com/xqgex/SecurityofCyberPhysicalSystems/blob/main/LICENSE
"""

from abc import ABC, abstractmethod
from enum import auto, Enum
from numbers import Number
from operator import add, eq, mul
from typing import Any, List, Optional, Tuple, Union

import attr
from attr.validators import deep_iterable, instance_of


class VectorOrientation(Enum):
    """ Enum for vector orientation (horizontal or vertical). """
    HORIZONTAL = auto()
    VERTICAL = auto()

    def __invert__(self) -> 'VectorOrientation':
        """
        >>> ~VectorOrientation.HORIZONTAL
        <VectorOrientation.VERTICAL>
        >>> ~VectorOrientation.VERTICAL
        <VectorOrientation.HORIZONTAL>
        """
        return next(k for k in self.__class__._member_map_.values() if k != self)  # pylint: disable=no-member

    def __repr__(self) -> str:
        return f'<{self}>'


class _Base(ABC):
    """ Base class for matrices and vectors. """

    def __add__(self, other: Union[Number, '_Base']) -> '_Base':
        return calculate_add_operator(self, other)

    @abstractmethod
    def __getitem__(self, index: int) -> Union[Number, '_Base']:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __mul__(self, other: Union[Number, '_Base']) -> '_Base':
        return calculate_mul_operator(self, other)

    @abstractmethod
    def __neg__(self) -> '_Base':
        raise NotImplementedError

    def __radd__(self, other: Union[Number, '_Base']) -> '_Base':
        return self.__add__(other)

    @abstractmethod
    def __repr__(self) -> 'str':
        raise NotImplementedError

    def __rmul__(self, other: Union[Number, '_Base']) -> '_Base':
        return calculate_mul_operator(other, self)

    def __rsub__(self, other: Union[Number, '_Base']) -> '_Base':
        return (-self).__add__(other)

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    def __sub__(self, other: Union[Number, '_Base']) -> '_Base':
        return self.__add__(-other)

    @abstractmethod
    def evolve(self, _: Any) -> '_Base':
        """ Wrapper for `attr.evolve`. """
        raise NotImplementedError

    @abstractmethod
    def expand_with_indices(self, indices: Tuple[int, ...], size: int) -> '_Base':
        """ Expand the instance to a new size. """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> Tuple[int, ...]:
        """ Return the instance dimensions. """
        raise NotImplementedError

    @property
    @abstractmethod
    def transpose(self) -> '_Base':
        """ Transpose the instance. """
        raise NotImplementedError


@attr.s(frozen=True, repr=False, slots=True)
class Vector(_Base):
    """ Implementation of a vector. """

    _vector: Tuple[float, ...] = attr.ib(
        validator=deep_iterable(instance_of(float), instance_of(tuple)),
        )
    orientation: VectorOrientation = attr.ib(
        default=VectorOrientation.VERTICAL,
        eq=False,
        validator=instance_of(VectorOrientation),
        )

    def __getitem__(self, key: int) -> float:
        return self._vector[key]

    def __iter__(self) -> float:
        for v in self._vector:
            yield v

    def __len__(self) -> int:
        return len(self._vector)

    def __neg__(self) -> 'Vector':
        return self.evolve(vector=tuple(-v for v in self._vector))

    def __repr__(self) -> 'str':
        return f'<{self.orientation.name.title()} vector of size {len(self)}>'

    def __str__(self) -> str:
        return str(list(self._vector))

    def evolve(self, vector: Tuple[float, ...]) -> 'Vector':
        return attr.evolve(self, vector=vector)

    def expand_with_indices(self, indices: Tuple[int, ...], size: int) -> 'Vector':
        """
        >>> print(Vector((1., 2.)).expand_with_indices((2, 4), 6))
        [0.0, 0.0, 1.0, 0.0, 2.0, 0.0]
        """
        if len(indices) != len(self):
            raise ValueError(f'Cannot expand vector with indices of different size.\n{self}\n{indices}')
        self_iterator = (v for v in self)
        return self.evolve(vector=tuple(next(self_iterator) if i in indices else 0.0 for i in range(size)))

    def filter_with_indices(self, indices: Tuple[int, ...]) -> 'Vector':
        """ Select the vector values that match the given indices and return a new vector from those values.

        >>> print(Vector((0., 1., 2., 3., 4., 5.)).filter_with_indices((1, 3, 4)))
        [1.0, 3.0, 4.0]
        """
        return self.evolve(vector=tuple(self[i] for i in indices))

    @property
    def dimensions(self) -> Tuple[int, int]:
        return (len(self), 1) if self.orientation == VectorOrientation.VERTICAL else (1, len(self))

    @property
    def transpose(self) -> 'Vector':
        return attr.evolve(self, orientation=~self.orientation)

    @classmethod
    def from_list(cls, l: Union[List[float], Tuple[float, ...]]) -> 'Vector':
        """ Create a new instance from a list of float numbers. """
        if not isinstance(l, (list, tuple)):
            raise ValueError(f'Unsupported input type, got `{l}` of type {type(l)}, expected a list or a tuple.')
        return cls(vector=tuple(l), orientation=VectorOrientation.HORIZONTAL)

    @classmethod
    def of_size(cls, size: int) -> 'Vector':
        """ Create a new instance with a given size filled with zeros. """
        if size <= 0:
            raise ValueError(f'Cannot create a vector with a non positive size {size}.')
        return cls(vector=tuple(0.0 for _ in range(size)))


@attr.s(frozen=True, repr=False, slots=True)
class Matrix(_Base):
    """ Implementation of a matrix. """

    _matrix: Tuple[Vector, ...] = attr.ib(
        validator=deep_iterable(instance_of(Vector), instance_of(tuple)),
        )

    def __getitem__(self, index: int) -> Vector:
        """ Return a row from the matrix. """
        return self._matrix[index]

    def __iter__(self) -> Vector:
        for row in self._matrix:
            yield row

    def __len__(self) -> int:
        """ Return how many rows there are in the matrix. """
        return len(self._matrix)

    def __neg__(self) -> 'Matrix':
        return self.evolve(matrix=tuple(-row for row in self._matrix))

    def __repr__(self) -> 'str':
        rows, cols = self.dimensions
        return f'<Matrix with {rows} rows and {cols} columns>'

    def __str__(self) -> str:
        rows_str = []
        for index, row in enumerate(self._matrix):
            loop_str = '[' if index == 0 else ' '
            loop_str += str(list(row))
            loop_str += ',' if index + 1 < len(self) else ']'
            rows_str.append(loop_str)
        return '\n'.join(rows_str)

    def __attrs_post_init__(self) -> None:
        row_lengths = tuple(map(len, self._matrix))
        if any(l != row_lengths[0] for l in row_lengths):
            raise ValueError(f'Unaligned matrix is prohibited\n{str(self)}')

    def column(self, index: int) -> Vector:
        """ Return the column vector at a specific index. """
        return Vector(tuple(r[index] for r in self._matrix), orientation=VectorOrientation.VERTICAL)

    def evolve(self, matrix: Tuple[Vector, ...]) -> 'Matrix':
        if self.__class__ != Matrix and len(matrix) != len(matrix[0]):
            return Matrix.from_vectors(matrix)  # Downcast child instance to Matrix base class
        return attr.evolve(self, matrix=matrix)

    def expand_with_indices(self, _1: Tuple[int, ...], _2: int) -> 'Matrix':
        raise ValueError(f'Cannot expand non square matrix.\n{self}')

    def is_zero_matrix(self) -> bool:
        """ Return True if all of the matrix values are 0 and False o.w. """
        return all(c == 0. for r in self for c in r)

    def row(self, index: int) -> Vector:
        """ Explicit function equivalent to `self[index]`. """
        return self[index]

    def to_vector(self) -> Vector:
        """ Convert a matrix to a vector. """
        rows_count, cols_count = self.dimensions
        if rows_count == 1:  # Horizontal vector
            return self.row(0)
        if cols_count == 1:  # Vertical vector
            return self.column(0)
        raise AttributeError(f'Cannot convert the matrix into a vector\n{self}')

    @_matrix.validator
    def _matrix_validator(self, _, matrix: Tuple[Vector, ...]):
        for row in matrix:
            if row.orientation == VectorOrientation.VERTICAL:
                raise ValueError(
                    f'A matrix is a tuple of `horizontal` vectors, got a `vertical` vector:\n{row}\n{matrix}')

    @property
    def dimensions(self) -> Tuple[int, int]:
        return len(self), len(self._matrix[0])

    @property
    def is_square(self) -> bool:
        """ Is matrix instance have the same numbers of rows and columns. """
        return eq(*self.dimensions)

    @property
    def transpose(self) -> 'Matrix':
        return self.evolve(matrix=tuple(map(lambda it: Vector.from_list(list(it)), zip(*self._matrix))))

    @classmethod
    def diagonal(
            cls,
            value: Union[float, Tuple[float, ...]],
            rows_count: int,
            optional_cols_count: Optional[int]=None
            ) -> 'Matrix':
        """ Create a `rows_count` by `optional_cols_count` matrix that have `value` on the matrix diagonal.

        Note, `value` can be an iterable of length smaller or greater than the length of the diagonal.

        >>> print(Matrix.diagonal(1.0, 3))
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]]
        >>> print(Matrix.diagonal((1.0, 2.0), 3, 4))
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 2.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]]
        >>> print(Matrix.diagonal((1.0, 2.0, 3.0, 4.0, 5.0), 3, 2))
        [[1.0, 0.0],
         [0.0, 2.0],
         [0.0, 0.0]]
        """
        for size in (rows_count, optional_cols_count):
            if size is not None and size <= 0:
                raise ValueError(f'Cannot create a matrix with a non positive size {size}.')
        cols_count = optional_cols_count if optional_cols_count is not None else rows_count
        values = tuple(value for _ in range(min(rows_count, cols_count))) if isinstance(value, float) else \
                 tuple(value) + tuple(0.0 for _ in range(min(rows_count, cols_count) - len(value)))
        return cls.from_lists([[values[i] if i == j else 0.0 for i in range(cols_count)] for j in range(rows_count)])

    @classmethod
    def from_vector(cls, vector: Vector) -> 'Matrix':
        """ Create a new instance from a single vector. """
        return cls.from_lists(list(map(list, zip(vector))) if vector.orientation == VectorOrientation.VERTICAL else \
                              [list(vector)])

    @classmethod
    def from_vectors(cls, vectors: Tuple[Vector, ...]) -> 'Matrix':
        """ Create a new instance from tuple of vectors. """
        return cls(vectors)

    @classmethod
    def from_lists(cls, lists: Union[List[List[float]], Tuple[Tuple[float, ...]]]) -> 'Matrix':
        """
        >>> print(Matrix.from_lists([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]
        """
        return cls(tuple(map(Vector.from_list, lists)))

    @classmethod
    def of_size(cls, rows_count: int, cols_count: Optional[int]=None) -> 'Matrix':
        """ Create a new instance with a given size filled with zeros.

        >>> print(Matrix.of_size(3))
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]]
        """
        return cls.diagonal(0.0, rows_count, cols_count)


def _calculate_add_two_matrices(first: Matrix, second: Matrix) -> Matrix:
    """
    >>> matrix_1 = Matrix.from_lists([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    >>> matrix_2 = Matrix.from_lists([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    >>> print(_calculate_add_two_matrices(matrix_1, matrix_2))
    [[6.0, 8.0, 10.0],
     [12.0, 14.0, 16.0]]
    """
    if first.dimensions != second.dimensions:
        raise TypeError(
            f'Cannot add two matrices of different sizes, self: {first.dimensions} other: {second.dimensions}')
    rows, cols = first.dimensions
    return first.evolve(matrix=tuple(Vector.from_list([add(first[r][c], second[r][c]) for c in range(cols)])
                                     for r in range(rows)))


def _calculate_add_two_vectors(first: Vector, second: Vector) -> Vector:
    """
    >>> print(_calculate_add_two_vectors(Vector.from_list([0.0, 1.0, 2.0]), Vector.from_list([3.0, 4.0, 5.0])))
    [3.0, 5.0, 7.0]
    """
    if first.dimensions != second.dimensions:
        raise TypeError(
            f'Cannot add two vectors of different sizes, self: {first.dimensions} other: {second.dimensions}')
    if first.orientation != second.orientation:
        raise TypeError(
            f'A {first.orientation.name.lower()} vector cannot be added to a {second.orientation.name.lower()} ' \
            f'vector.\n{first}\n{second}')
    return first.evolve(vector=tuple(map(add, first, second)))


def _calculate_multiply_two_matrices(first: Matrix, second: Matrix) -> Matrix:
    if first.dimensions[1] != second.dimensions[0]:
        f_dim = 'x'.join(map(str, first.dimensions))
        s_dim = 'x'.join(map(str, second.dimensions))
        raise TypeError(
            f'The product is not defined. Cannot multiply a {f_dim} matrix with {s_dim} matrix.\n{first}\n{second}')
    return first.evolve(matrix=tuple(Vector.from_list([sum(mul(a, b) for a, b in zip(first_r, second_c))
                                                       for second_c in zip(*second)])
                                     for first_r in first))


def _calculate_operator_with_a_number(
        first: Union[Matrix, Vector],
        second: Number,
        operator: Union[add, mul]
        ) -> Union[Matrix, Vector]:
    """
    >>> print(_calculate_operator_with_a_number(Vector.of_size(3), 10, add))
    [10.0, 10.0, 10.0]
    >>> print(_calculate_operator_with_a_number(Vector.of_size(3), 10, mul))
    [0.0, 0.0, 0.0]
    """
    if isinstance(first, Matrix):
        return first.evolve(matrix=tuple(Vector.from_list([operator(c ,second) for c in r]) for r in first))
    if isinstance(first, Vector):
        return first.evolve(vector=tuple(operator(c ,second) for c in first))
    raise TypeError(f'Unexpected input, cannot {operator} \'{type(first)}\' and \'{type(second)}\'')


def calculate_add_operator(
        first: Union[Matrix, Vector],
        second: Union[Number, Matrix, Vector]
        ) -> Union[Matrix, Vector]:
    """ Utility function shared between vectors and matrices to support additions. """
    if isinstance(second, Number):
        return _calculate_operator_with_a_number(first, second, add)
    if isinstance(first, Matrix) and isinstance(second, Matrix):
        return _calculate_add_two_matrices(first, second)
    if isinstance(first, Matrix) and 1 in first.dimensions:
        first = first.to_vector()
    if isinstance(second, Matrix) and 1 in second.dimensions:
        second = second.to_vector()
    if isinstance(first, Vector) and isinstance(second, Vector):
        return _calculate_add_two_vectors(first, second)
    raise TypeError(f'TypeError: unsupported operand type(s) for +: \'{type(first)}\' and \'{type(second)}\'')


def calculate_mul_operator(
        first: Union[Number, Matrix, Vector],
        second: Union[Number, Matrix, Vector]
        ) -> Union[Matrix, Vector]:
    """ Utility function shared between vectors and matrices to support multiplications. """
    if isinstance(first, Number):
        return _calculate_operator_with_a_number(second, first, mul)
    if isinstance(second, Number):
        return _calculate_operator_with_a_number(first, second, mul)
    first_matrix = Matrix.from_vector(first) if isinstance(first, Vector) else first
    second_matrix = Matrix.from_vector(second) if isinstance(second, Vector) else second
    return _calculate_multiply_two_matrices(first_matrix, second_matrix)
