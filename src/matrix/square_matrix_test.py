""" Tests for the square matrix class.

This code was proudly written using Gedit.
Released under the following license:
https://github.com/xqgex/SecurityofCyberPhysicalSystems/blob/main/LICENSE
"""

from typing import List

from hamcrest import assert_that, calling, equal_to, is_, raises
import pytest

from .square_matrix import SquareMatrix


class TestSquareMatrixClass:
    @pytest.mark.parametrize('invalid_matrix', [
        [[0.], [0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0.], [0., 0.], [0., 0.]],
        ])
    def test_that_creating_a_square_matrix_from_invalid_list_of_lists_raises_an_exception(
            self,
            invalid_matrix: List[List[float]],
            ) -> None:
        assert_that(calling(SquareMatrix.from_lists).with_args(invalid_matrix), raises(Exception))

    @pytest.mark.parametrize('matrix_1,matrix_2,expected', [
        (
            SquareMatrix.from_lists([[0., 1.], [2., 3.]]),
            SquareMatrix.from_lists([[4., 5.], [6., 7.]]),
            SquareMatrix.from_lists([[6., 7.], [26., 31.]]),
        ),
        (
            SquareMatrix.from_lists([[0., 0.], [0., 0.]]),
            SquareMatrix.from_lists([[1., 1.], [1., 1.]]),
            SquareMatrix.from_lists([[0., 0.], [0., 0.]]),
        ),
        ])
    def test_that_multiplying_two_square_matrices_is_as_expected(
            self,
            matrix_1: SquareMatrix,
            matrix_2: SquareMatrix,
            expected: SquareMatrix,
            ) -> None:
        assert_that(matrix_1 * matrix_2, is_(equal_to((expected))))

    @pytest.mark.parametrize('matrix_1,matrix_2,expected', [
        (
            SquareMatrix.from_lists([[0., 1.], [2., 3.]]),
            SquareMatrix.from_lists([[4., 5.], [6., 7.]]),
            SquareMatrix.from_lists([[4., 6.], [8., 10.]]),
        ),
        (
            SquareMatrix.from_lists([[0., 0.], [0., 0.]]),
            SquareMatrix.from_lists([[1., 1.], [1., 1.]]),
            SquareMatrix.from_lists([[1., 1.], [1., 1.]]),
        ),
        ])
    def test_that_adding_two_square_matrices_is_as_expected(
            self,
            matrix_1: SquareMatrix,
            matrix_2: SquareMatrix,
            expected: SquareMatrix,
            ) -> None:
        assert_that(matrix_1 + matrix_2, is_(equal_to((expected))))

    @pytest.mark.parametrize('matrix,expected', [
        (
            SquareMatrix.from_lists([[4., 7.], [2., 6.]]),
            SquareMatrix.from_lists([[0.6, -0.7], [-0.2, 0.4]]),
        ),
        (
            SquareMatrix.from_lists([[3., 3.5], [3.2, 3.6]]),
            SquareMatrix.from_lists([[-9., 8.75], [8., -7.5]]),
        ),
        ])
    def test_that_the_inverse_matrix_of_example_square_matrices_is_as_expected(
            self,
            matrix: SquareMatrix,
            expected: SquareMatrix,
            ) -> None:
        assert_that(matrix.inverse(), is_(equal_to((expected))))

    @pytest.mark.parametrize('invalid_matrix', [
        SquareMatrix.from_lists([[3., 4.], [6., 8.]]),
        ])
    def test_that_trying_to_inverse_a_matrix_that_have_no_inverse_raises_an_exception(
            self,
            invalid_matrix: SquareMatrix,
            ) -> None:
        assert_that(calling(invalid_matrix.inverse), raises(Exception))

    @pytest.mark.parametrize('matrix,expected', [
        (
            SquareMatrix.from_lists([[1., 2.], [2., 13.]]),
            SquareMatrix.from_lists([[1., 0.], [2., 3.]]),
        ),
        (
            SquareMatrix.from_lists([[64., 64., 64.], [64., 128., 128.], [64., 128., 192.]]),
            SquareMatrix.from_lists([[8., 0., 0.], [8., 8., 0.], [8., 8., 8.]]),
        ),
        (
            SquareMatrix.from_lists([[1.0, 2.0, 8.0, 64.0],
                                     [2.0, 20.0, 80.0, 640.0],
                                     [8.0, 80.0, 1344.0, 10752.0],
                                     [64.0, 640.0, 10752.0, 348160.0]]),
            SquareMatrix.from_lists([[1., 0., 0., 0.],
                                     [2., 4., 0., 0.],
                                     [8., 16., 32., 0.],
                                     [64., 128., 256., 512.]]),
        ),
        ])
    def test_that_the_square_root_of_example_square_matrices_is_the_expected_lower_triangular_matrix(
            self,
            matrix: SquareMatrix,
            expected: SquareMatrix,
            ) -> None:
        assert_that(matrix.square_root(), is_(equal_to((expected))))

    @pytest.mark.parametrize('matrix', [
        SquareMatrix.from_lists([[1., 2.], [2., 4.]]),
        SquareMatrix.from_lists([[36., 30., 18.], [30., 41., 23.], [18., 23., 14.]]),
        SquareMatrix.from_lists([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]]),
        ])
    def test_that_multiplying_a_square_root_of_a_matrix_with_tranpose_of_itself_returns_the_original_matrix(
            self,
            matrix: SquareMatrix,
            ) -> None:
        square_root = matrix.square_root()
        assert_that(square_root * square_root.transpose, is_(equal_to((matrix))))

    @pytest.mark.parametrize('invalid_matrix', [
        SquareMatrix.from_lists([[1., 2.], [3., 0.]]),
        SquareMatrix.from_lists([[0., 2.], [3., 4.]]),
        SquareMatrix.from_lists([[1., 0., 0.], [0., 1., 0.], [0., 1., 0.]]),
        SquareMatrix.from_lists([[1., 1., 1.], [1., 1., 1.], [1., 1., 0.]]),
        ])
    def test_that_trying_to_get_square_root_for_a_matrix_with_zero_on_the_diagonal_raises_an_exception(
            self,
            invalid_matrix: SquareMatrix,
            ) -> None:
        assert_that(calling(invalid_matrix.square_root), raises(Exception))
