""" Testing data.

This code was proudly written using Gedit.
Released under the following license:
https://github.com/xqgex/SecurityofCyberPhysicalSystems/blob/main/LICENSE
"""

from collections import namedtuple
from csv import DictReader
from itertools import islice
from math import cos
from typing import Optional, Tuple

_Coordinate = namedtuple('_Coordinate', ['latitude', 'longitude'])

# See the link in the README file for more information about the data set.
_CSV_FILE_NAME = './test_data/PVS_1_dataset_gps_mpu_left.csv'
_CSV_HEADER_ACCELERATION_X = 'acc_x_dashboard'
_CSV_HEADER_ACCELERATION_Y = 'acc_y_dashboard'
_CSV_HEADER_LATITUDE = 'latitude'
_CSV_HEADER_LONGITUDE = 'longitude'
_CSV_HEADER_TIMESTAMP = 'timestamp'
_CSV_OPEN_ENCODING = 'utf-8'
_CSV_READ_MODE = 'r'
_EARTH_RADIUS_KM = 6371
_IMAGE_SIZE_METER = 2800
_MAP_CORNER_BOTTOM_LEFT = _Coordinate(latitude=-27.720931, longitude=-51.143431)
_MAP_CORNER_TOP_RIGHT = _Coordinate(latitude=-27.680405, longitude=-51.097423)
_TIMESTAMP_PRECISION = 2


class _Point:  # pylint: disable=too-few-public-methods
    """ A simple utility class for an (x,y) point. """

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    @classmethod
    def from_coordinate(cls, coordinate: _Coordinate) -> '_Point':
        """ Create an (x,y) point from earth coordinate `latitude,longitude`. """
        x_average = (_MAP_CORNER_TOP_RIGHT.latitude + _MAP_CORNER_BOTTOM_LEFT.latitude) / 2
        x = _EARTH_RADIUS_KM * coordinate.longitude * cos(x_average)
        y = _EARTH_RADIUS_KM * coordinate.latitude
        return cls(x, y)


class _RelativeXY:  # pylint: disable=too-few-public-methods
    def __init__(self) -> None:
        self._earth_bottom_right = _Point.from_coordinate(_MAP_CORNER_BOTTOM_LEFT)
        self._screen_bottom_right = _Point(x=0.0, y=0.0)
        self._earth_top_left = _Point.from_coordinate(_MAP_CORNER_TOP_RIGHT)
        self._screen_top_left = _Point(x=_IMAGE_SIZE_METER, y=_IMAGE_SIZE_METER)

    def x_y_for_earth_coordinate(self, coordinate: _Coordinate) -> _Point:
        """ Get a relative (x,y) point for a given earth coordinate. """
        point = _Point.from_coordinate(coordinate)
        screen_x = self._screen_top_left.x + self._ratio_width * (point.x - self._earth_top_left.x)
        screen_y = self._screen_top_left.y + self._ratio_height * (point.y - self._earth_top_left.y)
        return _Point(x=screen_x, y=screen_y)

    @property
    def _earth_height(self) -> float:
        return self._earth_top_left.y - self._earth_bottom_right.y

    @property
    def _earth_width(self) -> float:
        return self._earth_top_left.x - self._earth_bottom_right.x

    @property
    def _ratio_height(self) -> float:
        return self._screen_height / self._earth_height

    @property
    def _ratio_width(self) -> float:
        return self._screen_width / self._earth_width

    @property
    def _screen_height(self) -> float:
        return self._screen_top_left.y - self._screen_bottom_right.y

    @property
    def _screen_width(self) -> float:
        return self._screen_top_left.x - self._screen_bottom_right.x


def get_test_data(
        measurements_to_skip: int=0,
        max_measurements_to_load: Optional[int]=None,
        ) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    """ Retrieve the testing data.

    The function returns 5 tuples of floats, matching the following data:
    1. Timestamps (starts at t=0)
    2. Relative X position
    3. Relative Y position
    4. Acceleration X axis
    5. Acceleration Y axis

    :param int measurements_to_skip: If greater than 0, the first selected number of measurements will be skipped.
    :param Optional[int] max_measurements_to_load: If set, the maximum number of measurements will be limited.
    :returns: The data from the dataset.
    :rtype: Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]
    """
    relative_world = _RelativeXY()
    acceleration_x = []
    acceleration_y = []
    relative_x = []
    relative_y = []
    timestamps = []
    timestamp_t_0 = None
    with open(_CSV_FILE_NAME, encoding=_CSV_OPEN_ENCODING, mode=_CSV_READ_MODE) as csvfile:
        reader = DictReader(csvfile)
        for index, row in enumerate(islice(reader, measurements_to_skip, None)):
            if max_measurements_to_load is not None and max_measurements_to_load <= index:
                break
            acceleration_x.append(float(row[_CSV_HEADER_ACCELERATION_X]))
            acceleration_y.append(float(row[_CSV_HEADER_ACCELERATION_Y]))
            if timestamp_t_0 is None:
                timestamp_t_0 = float(row[_CSV_HEADER_TIMESTAMP])
            timestamps.append(round(float(row[_CSV_HEADER_TIMESTAMP]) - timestamp_t_0, _TIMESTAMP_PRECISION))
            coordinate = _Coordinate(float(row[_CSV_HEADER_LATITUDE]), float(row[_CSV_HEADER_LONGITUDE]))
            point = relative_world.x_y_for_earth_coordinate(coordinate)
            relative_x.append(point.x)
            relative_y.append(point.y)
    return timestamps, relative_x, relative_y, acceleration_x, acceleration_y
