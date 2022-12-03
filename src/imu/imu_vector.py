""" A simple implementation of an IMU vector.

This code was proudly written using Gedit.
Released under the following license:
https://github.com/xqgex/SecurityofCyberPhysicalSystems/blob/main/LICENSE
"""

from functools import cached_property
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import attr
from attr.validators import ge, instance_of, le, optional

from matrix import SquareMatrix, Vector, VectorOrientation

_CHAR_UNDERSCORE = '_'
_MEASUREMENT_PREFIX = 'measurement_noise_'


class Noise(NamedTuple):
    """ Named tuple to group together the measurement noise and the process noise. """
    measurement: float
    process: float


_NOISE = {
    'x': Noise(measurement=1., process=1.),
    'y': Noise(measurement=1., process=1.),
    'v_x': Noise(measurement=1., process=1.),
    'v_y': Noise(measurement=1., process=1.),
    'a_x': Noise(measurement=1., process=1.),
    'a_y': Noise(measurement=1., process=1.),
    }


@attr.s(frozen=True, kw_only=True, slots=True)
class IMUVector:
    """ Position vector.

    :math:`imu_vector = IMUVector(x,y,θ,v,a,ω)`
    where:
    * `x` [m]: X position
    * `y` [m]: Y position
    * `v_x` [m/s]: Velocity in X axis
    * `v_y` [m/s]: Velocity in Y axis
    * `a_x` [m/s^2]: Acceleration in X axis (a = dv / dt)
    * `a_y` [m/s^2]: Acceleration in Y axis (a = dv / dt)
    """

    __dict__: Dict[str, Any] = attr.ib(
        eq=False,
        factory=dict,
        init=False,
        metadata={'class': 'IMUVector'},
        repr=False,
        ) # To support cached properties <https://github.com/python-attrs/attrs/issues/164#issuecomment-1233241377>.
    x: Optional[float] = attr.ib(
        default=None,
        validator=optional(instance_of(float)),
        )  # pylint: disable=invalid-name
    y: Optional[float] = attr.ib(
        default=None,
        validator=optional(instance_of(float)),
        )  # pylint: disable=invalid-name
    v_x: Optional[float] = attr.ib(
        default=None,
        validator=optional([instance_of(float), ge(-100.0), le(100.0)]),
        )
    v_y: Optional[float] = attr.ib(
        default=None,
        validator=optional([instance_of(float), ge(-100.0), le(100.0)]),
        )
    a_x: Optional[float] = attr.ib(
        default=None,
        validator=optional([instance_of(float), ge(-35.0), le(35.0)]),
        )
    a_y: Optional[float] = attr.ib(
        default=None,
        validator=optional([instance_of(float), ge(-35.0), le(35.0)]),
        )
    measurement_noise_x: float = attr.ib(
        default=_NOISE['x'].measurement,
        repr=False,
        validator=instance_of(float),
        )
    measurement_noise_y: float = attr.ib(
        default=_NOISE['y'].measurement,
        repr=False,
        validator=instance_of(float),
        )
    measurement_noise_v_x: float = attr.ib(
        default=_NOISE['v_x'].measurement,
        repr=False,
        validator=instance_of(float),
        )
    measurement_noise_v_y: float = attr.ib(
        default=_NOISE['v_y'].measurement,
        repr=False,
        validator=instance_of(float),
        )
    measurement_noise_a_x: float = attr.ib(
        default=_NOISE['a_x'].measurement,
        repr=False,
        validator=instance_of(float),
        )
    measurement_noise_a_y: float = attr.ib(
        default=_NOISE['a_y'].measurement,
        repr=False,
        validator=instance_of(float),
        )

    def __add__(self, other: Union[int, float, 'IMUVector', Vector]) -> 'IMUVector':
        """
        >>> IMUVector(x=1.2, v_y=0.0) + 2
        IMUVector(x=3.2, y=None, v_x=None, v_y=2.0, a_x=None, a_y=None)
        >>> IMUVector(x=1.2) + IMUVector(y=2.4)
        IMUVector(x=1.2, y=2.4, v_x=None, v_y=None, a_x=None, a_y=None)
        >>> IMUVector(x=2.0, y=None, v_x=6.0, v_y=None, a_x=None, a_y=None) + Vector.from_list([1., 2.])
        IMUVector(x=3.0, y=None, v_x=8.0, v_y=None, a_x=None, a_y=None)
        """
        def _add_dicts(dict_1: Dict[str, float], dict_2: Dict[str, float]) -> Dict[str, float]:
            return {k: dict_1.get(k, 0.0) + dict_2.get(k, 0.0) for k in set(list(dict_1.keys()) + list(dict_2.keys()))}
        if isinstance(other, IMUVector):
            return attr.evolve(self, **_add_dicts(self.to_dict(), other.to_dict()))
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError(
                    'Cannot perform arithmetic operation between `IMUVector` and `Vector` of different sizes.\n' \
                    f'{self}\n{other}')
            return attr.evolve(self, **{k: v + other[i] for i, (k, v) in enumerate(self.to_dict().items())})
        return attr.evolve(self, **{k: v + other for k, v in self.to_dict().items()})

    def __len__(self) -> int:
        """
        >>> len(IMUVector(x=1.2))
        1
        """
        return len(self.to_dict())

    def __mul__(self, other: Union[int, float, 'IMUVector']) -> 'IMUVector':
        """
        >>> IMUVector(x=1.2) * 2
        IMUVector(x=2.4, y=None, v_x=None, v_y=None, a_x=None, a_y=None)
        >>> print(IMUVector(x=1.0, y=2.0, v_y=3.0, a_x=4.0) * IMUVector(x=5.0, y=6.0, v_x=7.0, a_x=8.0))
        IMUVector(x=5.0, y=12.0, v_x=0.0, v_y=0.0, a_x=32.0, a_y=None)
        """
        def _multiply_dicts(dict_1: Dict[str, float], dict_2: Dict[str, float]) -> Dict[str, float]:
            return {k: dict_1.get(k, 0.0) * dict_2.get(k, 0.0) for k in set(list(dict_1.keys()) + list(dict_2.keys()))}
        if isinstance(other, IMUVector):
            return attr.evolve(self, **_multiply_dicts(self.to_dict(), other.to_dict()))
        return attr.evolve(self, **{k: v * other for k, v in self.to_dict().items()})

    def __neg__(self) -> 'IMUVector':
        """
        >>> -IMUVector(x=1.2, y=0.0)
        IMUVector(x=-1.2, y=-0.0, v_x=None, v_y=None, a_x=None, a_y=None)
        """
        return attr.evolve(self, **{k: -v for k, v in self.to_dict().items()})

    def __radd__(self, other: Union[int, float, 'IMUVector']) -> 'IMUVector':
        """
        >>> imu_vector = IMUVector(x=1.2)
        >>> 1.0 + imu_vector == imu_vector + 1.0
        True
        """
        return self.__add__(other)

    def __rmul__(self, other: Union[int, float, 'IMUVector']) -> 'IMUVector':
        """
        >>> imu_vector = IMUVector(x=1.2)
        >>> 2 * imu_vector == imu_vector * 2
        True
        """
        return self.__mul__(other)

    def __rsub__(self, other: Union[int, float, 'IMUVector']) -> 'IMUVector':
        """
        >>> 2 - IMUVector(x=1.2, a_x=2.0)
        IMUVector(x=0.8, y=None, v_x=None, v_y=None, a_x=0.0, a_y=None)
        """
        return (-self).__add__(other)

    def __sub__(self, other: Union[int, float, 'IMUVector']) -> 'IMUVector':
        """
        >>> IMUVector(v_x=5.0, a_y=5.0) - 2
        IMUVector(x=None, y=None, v_x=3.0, v_y=None, a_x=None, a_y=3.0)
        """
        return self.__add__(-other)

    def to_dict(self) -> Dict[str, float]:
        """
        >>> IMUVector(v_x=10.0, a_y=11.0).to_dict()
        {'v_x': 10.0, 'a_y': 11.0}
        """
        return {v[0]: v[1] for v in self._index_to_member_map_.values() if v is not None}

    def to_minimal_vector(self) -> Vector:
        """
        >>> str(IMUVector(x=1.0, v_x=2.0, a_y=3.0).to_minimal_vector())
        '[1.0, 2.0, 3.0]'
        >>> len(IMUVector(x=1.0, v_x=2.0, a_y=3.0).to_minimal_vector())
        3
        """
        return Vector(tuple(v[1] for v in self._index_to_member_map_.values() if v is not None),
                      orientation=VectorOrientation.VERTICAL)

    @cached_property
    def _index_to_member_map_(self) -> Dict[int, Optional[Tuple[str, float]]]:
        """
        >>> IMUVector(v_x=10.0, a_y=11.0)._index_to_member_map_
        {0: None, 1: None, 2: ('v_x', 10.0), 3: None, 4: None, 5: ('a_y', 11.0)}
        """
        # pylint: disable-next=unnecessary-dunder-call
        return {i: (f, self.__getattribute__(f)) if self.__getattribute__(f) is not None else None
                for i, f in enumerate(self.__class__.fields())}

    @cached_property
    def indices(self) -> Tuple[int, ...]:
        """
        >>> IMUVector(x=1.0, v_x=1.0, a_y=1.0).indices
        (0, 2, 5)
        """
        return tuple(k for k, v in self._index_to_member_map_.items() if v is not None)

    @cached_property
    def measurement_noise_R(self) -> SquareMatrix:
        """ Return the IMU Vector measurement noise (R). """
        # pylint: disable-next=unnecessary-dunder-call
        diagonal_values = tuple(self.__getattribute__(_MEASUREMENT_PREFIX + v[0])
                                for v in self._index_to_member_map_.values() if v is not None)
        return SquareMatrix.diagonal(diagonal_values, len(diagonal_values))

    @classmethod
    def fields(cls) -> Tuple[str, ...]:
        """
        >>> IMUVector.fields()
        ('x', 'y', 'v_x', 'v_y', 'a_x', 'a_y')
        >>> len(IMUVector.fields())
        6
        """
        def _is_imu_vector_field(a) -> bool:
            return not a.name.startswith(_MEASUREMENT_PREFIX) and not a.name.startswith(_CHAR_UNDERSCORE)
        return tuple(a.name for a in cls.__attrs_attrs__ if _is_imu_vector_field(a))

    @classmethod
    def from_vector(cls, vector: Vector) -> 'IMUVector':
        """
        >>> IMUVector.from_vector(Vector((0.0, 1.1, 2.2, 3.3, 4.4, 5.5)))
        IMUVector(x=0.0, y=1.1, v_x=2.2, v_y=3.3, a_x=4.4, a_y=5.5)
        """
        if len(vector) != len(IMUVector.fields()):
            raise ValueError(f'Cannot create IMU vector from the input vector, missing fields, got: {vector}')
        return cls(**{a: vector[i] for i, a in enumerate(IMUVector.fields())})

    @classmethod
    def function_f(cls, vector: Vector, process_noise_Q: SquareMatrix, time_delta: float) -> Vector:
        """ Function :math:`F()` used for prediction based on time delta and the process noise.

        .. math::
           x - x_0 = v_0^{(x)} * t + 0.5 * a_0^{(x)} * t^2 \\
           y - y_0 = v_0^{(y)} * t + 0.5 * a_0^{(y)} * t^2 \\
           v^{(x)} - v_0^{(x)} = a_0^{(x)} * t \\
           v^{(y)} - v_0^{(y)} = a_0^{(y)} * t
        """
        values_update = (
            vector[2] * time_delta + 0.5 * vector[4] * pow(time_delta, 2),
            vector[3] * time_delta + 0.5 * vector[5] * pow(time_delta, 2),
            vector[4] * time_delta,
            vector[5] * time_delta,
            0.0,
            0.0,
            )
        return vector + (process_noise_Q * Vector(values_update, orientation=VectorOrientation.VERTICAL))

    @classmethod
    def function_h(cls, vector: Vector, measurement_noise_R: SquareMatrix) -> Vector:
        """ Function :math:`H()` is updating the prediction with the the measurements noise. """
        return (measurement_noise_R * vector).column(0)

    @classmethod
    def process_noise_Q(cls) -> SquareMatrix:
        """ Return the IMU Vector process noise (Q). """
        return SquareMatrix.diagonal(tuple(_NOISE[f].process for f in IMUVector.fields()), len(IMUVector.fields()))


# In order for the @cached_property Doctests to be discovered.
# <https://stackoverflow.com/a/72500890/4678126>
# pylint: disable=protected-access
__test__= {
    'IMUVector._index_to_member_map_': IMUVector._index_to_member_map_,
    'IMUVector.indices': IMUVector.indices,
    'IMUVector.measurement_noise_R': IMUVector.measurement_noise_R,
    }
# pylint: enable=protected-access
