""" Tests for the matrix class.

This code was proudly written using Gedit.
Released under the following license:
https://github.com/xqgex/SecurityofCyberPhysicalSystems/blob/main/LICENSE
"""

from typing import Any, List

from hamcrest import assert_that, calling, equal_to, is_, not_, raises
import pytest

from .data_types import Matrix, Vector, VectorOrientation


class TestMatrixClass:
    @pytest.mark.parametrize('matrix', [
        [],
        [[0.]],
        [[0., 0.], [0., 0.]],
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        ])
    def test_that_a_matrix_can_be_created_from_a_list_of_lists(
            self,
            matrix: List[List[float]],
            ) -> None:
        assert_that(calling(Matrix.from_lists).with_args(matrix), not_(raises(Exception)))

    @pytest.mark.parametrize('size', range(1, 10))
    def test_that_a_matrix_can_be_created_with_a_positive_integer_size(self,size: int) -> None:
        assert_that(calling(Matrix.of_size).with_args(size), not_(raises(Exception)))

    @pytest.mark.parametrize('invalid_matrix', [
        [[0.], [0., 0.]],
        [[0., 0.], [0., 0., 0.], [0., 0.]],
        [[0., 0., 0.], [0., 0.], [0., 0., 0.]],
        ])
    def test_that_creating_a_matrix_from_invalid_list_of_lists_raises_an_exception(
            self,
            invalid_matrix: List[List[float]],
            ) -> None:
        assert_that(calling(Matrix.from_lists).with_args(invalid_matrix), raises(Exception))

    @pytest.mark.parametrize('non_positive_size', range(-10, 1))
    def test_that_creating_a_matrix_with_a_non_positive_size_raises_an_exception(self,non_positive_size: int) -> None:
        assert_that(calling(Matrix.of_size).with_args(non_positive_size), raises(Exception))

    @pytest.mark.parametrize('vector,expected', [
        (
            Vector.from_list([3., 4., 5.]),
            Matrix.from_lists([[3., 4., 5.]]),
        ),
        (
            Vector.from_list([3., 4., 5.]).transpose,
            Matrix.from_lists([[3.], [4.], [5.]]),
        ),
        (
            Vector((3., 4., 5.), orientation=VectorOrientation.VERTICAL),
            Matrix.from_lists([[3.], [4.], [5.]]),
        ),
        ])
    def test_that_creating_a_matrix_from_vector_is_as_expected(
            self,
            vector: Vector,
            expected: Matrix
            ) -> None:
        assert_that(Matrix.from_vector(vector), is_(equal_to((expected))))

    @pytest.mark.parametrize('matrix,scalar,expected', [
        (Matrix.from_lists([[9.]]), 3, Matrix.from_lists([[27.]])),
        (Matrix.from_lists([[1., 1.], [1., 1.]]), -5, Matrix.from_lists([[-5., -5.], [-5., -5.]])),
        (
            Matrix.from_lists([[-4., -3., -2.], [-1., 0., 1.], [2., 3., 4.]]),
            -1,
            Matrix.from_lists([[4., 3., 2.], [1., 0., -1.], [-2., -3., -4.]])
        ),
        (Matrix.from_lists([[1., 5.], [2., 6.]]), 3., Matrix.from_lists([[3., 15.], [6., 18.]])),
        (Matrix.from_lists([[-1., 0.], [1., 2.]]), 0, Matrix.from_lists([[0., 0.], [0., 0.]])),
        (Matrix.from_lists([[1., 1.5], [2., 2.5]]), 1.5, Matrix.from_lists([[1.5, 2.25], [3., 3.75]])),
        ])
    def test_that_multiplying_a_matrix_with_a_scalar_is_as_expected(
            self,
            matrix: Matrix,
            scalar: Any,
            expected: Matrix,
            ) -> None:
        assert_that(scalar * matrix, is_(equal_to((expected))))

    @pytest.mark.parametrize('matrix_1,matrix_2,expected', [
        (
            Matrix.from_lists([[1., 2., 3.], [-1., 1., 2.], [2., -3., 4.]]),
            Matrix.from_lists([[2., 1.], [-1., 2.], [3., 3.]]),
            Matrix.from_lists([[9., 14.], [3., 7.], [19., 8.]]),
        ),
        (
            Matrix.from_lists([[-1., 0., 1.], [-1., -1., 3.], [2., -2., 2.]]),
            Matrix.from_lists([[4., 1.], [-3., 2.], [1., -3.]]),
            Matrix.from_lists([[-3., -4.], [2., -12.], [16., -8.]]),
        ),
        (
            Matrix.from_lists([[2., 3., 4., 1.]]),
            Matrix.from_lists([[-1.], [3.], [-2.], [1.]]),
            Matrix.from_lists([[0.]]),
        ),
        (
            Matrix.from_lists([[3., 3., 4., 1.]]),
            Matrix.from_lists([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
            Matrix.from_lists([[3., 3., 4., 1.]]),
        ),
        (
            Matrix.from_lists([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
            Matrix.from_lists([[-1., 1., -1., 2.], [-4., -1., 3., -3.], [2., -3., 1., 4.], [3., -1., -2., 3.]]),
            Matrix.from_lists([[-1., 1., -1., 2.], [-4., -1., 3., -3.], [2., -3., 1., 4.], [3., -1., -2., 3.]]),
        ),
        (
            Matrix.from_lists([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
            Matrix.from_lists([[-1.], [3.], [-2.], [1.]]),
            Matrix.from_lists([[-1.], [3.], [-2.], [1.]]),
        ),
        (
            Matrix.from_lists([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]]),
            Matrix.from_lists([[-2.], [1.], [0.]]),
            Matrix.from_lists([[0.], [-3.], [-6.], [-9.]]),
        ),
        (
            Matrix.from_lists([[1., 2., 3.], [4., 5., 6.]]),
            Matrix.from_lists([[1., 2.], [3., 4.], [5., 6.]]),
            Matrix.from_lists([[22., 28.], [49., 64.]]),
        ),
        (
            Matrix.from_lists([[1., 2.], [3., 4.], [5., 6.]]),
            Matrix.from_lists([[1., 2., 3.], [4., 5., 6.]]),
            Matrix.from_lists([[9., 12., 15.], [19., 26., 33.], [29., 40., 51.]]),
        ),
        ])
    def test_that_multiplying_two_matrices_is_as_expected(
            self,
            matrix_1: Matrix,
            matrix_2: Matrix,
            expected: Matrix,
            ) -> None:
        assert_that(matrix_1 * matrix_2, is_(equal_to((expected))))

    @pytest.mark.parametrize('matrix_1,matrix_2', [
        (
            Matrix.from_lists([[-1., 0., 1.], [-1., -1., 3.], [2., -2., 2.]]),
            Matrix.from_lists([[2.], [7.]]),
        ),
        (
            Matrix.from_lists([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]]),
            Matrix.from_lists([[-3.], [-2.], [-1.], [0.]]),
        ),
        ])
    def test_that_multiplying_two_matrices_with_wrong_sizes_raises_an_exception(
            self,
            matrix_1: Matrix,
            matrix_2: Matrix,
            ) -> None:
        assert_that(calling(matrix_1.__mul__).with_args(matrix_2), raises(Exception))

    @pytest.mark.parametrize('matrix,scalar,expected', [
        (Matrix.from_lists([[0., 0.], [0., 0.]]), 0, Matrix.from_lists([[0., 0.], [0., 0.]])),
        (Matrix.from_lists([[0., 0.], [0., 0.]]), 1, Matrix.from_lists([[1., 1.], [1., 1.]])),
        (Matrix.from_lists([[0., 0.], [0., 0.]]), -1, Matrix.from_lists([[-1., -1.], [-1., -1.]])),
        ])
    def test_that_adding_a_scalar_to_a_matrix_is_as_expected(
            self,
            matrix: Matrix,
            scalar: Any,
            expected: Matrix,
            ) -> None:
        assert_that(scalar + matrix, is_(equal_to((expected))))

    @pytest.mark.parametrize('matrix_1,matrix_2,expected', [
        (
            Matrix.from_lists([[0., 1.], [2., 3.]]),
            Matrix.from_lists([[4., 5.], [6., 7.]]),
            Matrix.from_lists([[4., 6.], [8., 10.]])
        ),
        (
            Matrix.from_lists([[0., 0.], [0., 0.]]),
            Matrix.from_lists([[1., 1.], [1., 1.]]),
            Matrix.from_lists([[1., 1.], [1., 1.]]),
        ),
        ])
    def test_that_adding_two_matrices_is_as_expected(
            self,
            matrix_1: Matrix,
            matrix_2: Matrix,
            expected: Matrix,
            ) -> None:
        assert_that(matrix_1 + matrix_2, is_(equal_to((expected))))

    @pytest.mark.parametrize('matrix,expected', [
        (
            Matrix.from_lists([[0., 1.], [2., 3.]]),
            Matrix.from_lists([[0., 2.], [1., 3.]])
        ),
        (
            Matrix.from_lists([[1., 2., 3.], [4., 5., 6.]]),
            Matrix.from_lists([[1., 4.], [2., 5.], [3., 6.]]),
        ),
        (
            Matrix.from_lists([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]),
            Matrix.from_lists([[1., 4., 7.], [2., 5., 8.], [3., 6., 9.]]),
        ),
        ])
    def test_that_transposing_a_matrix_is_as_expected(
            self,
            matrix: Matrix,
            expected: Matrix,
            ) -> None:
        assert_that(matrix.transpose, is_(equal_to((expected))))
