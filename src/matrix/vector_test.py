""" Tests for the vector class.

This code was proudly written using Gedit.
Released under the following license:
https://github.com/xqgex/SecurityofCyberPhysicalSystems/blob/main/LICENSE
"""

from operator import add, mul
from typing import Any, List

from hamcrest import assert_that, calling, equal_to, is_, not_, raises
import pytest

from .data_types import Matrix, Vector


class TestVectorClass:
    @pytest.mark.parametrize('vector', [
        [0.],
        [0., 0.],
        [0., 0., 0.],
        ])
    def test_that_a_vector_can_be_created_from_a_list(
            self,
            vector: List[float],
            ) -> None:
        assert_that(calling(Vector.from_list).with_args(vector), not_(raises(Exception)))

    @pytest.mark.parametrize('size', range(1, 10))
    def test_that_a_vector_can_be_created_with_a_positive_integer_size(self,size: int) -> None:
        assert_that(calling(Vector.of_size).with_args(size), not_(raises(Exception)))

    @pytest.mark.parametrize('negative_size', range(-10, 1))
    def test_that_creating_a_vector_with_a_non_positive_size_raises_an_exception(self,negative_size: int) -> None:
        assert_that(calling(Vector.of_size).with_args(negative_size), raises(Exception))

    @pytest.mark.parametrize('vector', [
        Vector.from_list([]),
        Vector.from_list([0.]),
        Vector.from_list([-1., 0., 1., 2.]),
        ])
    def test_that_applying_transpose_once_on_a_vector_returns_a_vector_that_was_only_been_transposed(
            self,
            vector: Vector,
            ) -> None:
        new_vector = vector.transpose
        assert_that(new_vector, is_(equal_to((vector))))
        assert_that(new_vector.orientation, is_(not_(equal_to((vector.orientation)))))

    @pytest.mark.parametrize('vector', [
        Vector.from_list([]),
        Vector.from_list([0.]),
        Vector.from_list([-1., 0., 1., 2.]),
        ])
    def test_that_applying_transpose_twice_on_a_vector_returns_a_vector_equals_to_the_original_vector(
            self,
            vector: Vector,
            ) -> None:
        new_vector = vector.transpose.transpose
        assert_that(new_vector, is_(equal_to((vector))))
        assert_that(new_vector.orientation, is_(equal_to((vector.orientation))))

    @pytest.mark.parametrize('vector,scalar,expected', [
        (Vector.from_list([9.]), 3, Vector.from_list([27.])),
        (Vector.from_list([1., 1.5, -1., 0.]), 1.5, Vector.from_list([1.5, 2.25, -1.5, 0.])),
        ])
    def test_that_multiplying_a_vector_with_a_scalar_is_as_expected(
            self,
            vector: Vector,
            scalar: Any,
            expected: Vector,
            ) -> None:
        assert_that(scalar * vector, is_(equal_to((expected))))

    @pytest.mark.parametrize('vector_1,vector_2,expected', [
        (
            Vector.from_list([0., 1., 2.]),
            Vector.from_list([3., 4., 5.]).transpose,
            Matrix.from_lists([[14.]]),
        ),
        (
            Vector.from_list([0., 1., 2.]).transpose,
            Vector.from_list([3., 4., 5.]),
            Matrix.from_lists([[0., 0., 0.], [3., 4., 5.], [6., 8., 10.]]),
        ),
        ])
    def test_that_multiplying_two_vectors_is_as_expected(
            self,
            vector_1: Vector,
            vector_2: Vector,
            expected: Vector,
            ) -> None:
        assert_that(vector_1 * vector_2, is_(equal_to((expected))))

    @pytest.mark.parametrize('vector_1,vector_2', [
        (
            Vector.from_list([0., 0.]),
            Vector.from_list([0., 0., 0.]),
        ),
        (
            Vector.from_list([0., 0., 0.]),
            Vector.from_list([0., 0.]),
        ),
        ])
    def test_that_multiplying_two_vectors_of_different_size_raises_an_exception(
            self,
            vector_1: Vector,
            vector_2: Vector,
            ) -> None:
        assert_that(calling(mul).with_args(vector_1, vector_2), raises(Exception))

    @pytest.mark.parametrize('vector,scalar,expected', [
        (Vector.from_list([0., 0.]), 0, Vector.from_list([0., 0.])),
        (Vector.from_list([0., 0.]), 1, Vector.from_list([1., 1.])),
        (Vector.from_list([0., 0.]), -1, Vector.from_list([-1., -1.])),
        ])
    def test_that_adding_a_scalar_to_a_vector_is_as_expected(
            self,
            vector: Vector,
            scalar: Any,
            expected: Vector,
            ) -> None:
        assert_that(scalar + vector, is_(equal_to((expected))))

    @pytest.mark.parametrize('vector_1,vector_2,expected', [
        (Vector.from_list([-1., 1.]), Vector.from_list([4., 5.]), Vector.from_list([3., 6.])),
        (Vector.from_list([0., 0.]), Vector.from_list([1., 1.]), Vector.from_list([1., 1.])),
        ])
    def test_that_adding_two_vectors_is_as_expected(
            self,
            vector_1: Vector,
            vector_2: Vector,
            expected: Vector,
            ) -> None:
        assert_that(vector_1 + vector_2, is_(equal_to((expected))))

    @pytest.mark.parametrize('vector_1,vector_2', [
        (
            Vector.from_list([0., 0., 0.]),
            Vector.from_list([0., 0., 0.]).transpose,
        ),
        (
            Vector.from_list([0., 0.]),
            Vector.from_list([0., 0., 0.]),
        ),
        (
            Vector.from_list([0., 0., 0.]),
            Vector.from_list([0., 0.]),
        ),
        ])
    def test_that_adding_two_vectors_of_different_size_raises_an_exception(
            self,
            vector_1: Vector,
            vector_2: Vector,
            ) -> None:
        assert_that(calling(add).with_args(vector_1, vector_2), raises(Exception))
