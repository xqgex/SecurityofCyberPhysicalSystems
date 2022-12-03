""" The entry point of the project.

This code was proudly written using Gedit.
Released under the following license:
https://github.com/xqgex/SecurityofCyberPhysicalSystems/blob/main/LICENSE
"""

from csv import DictReader, DictWriter
from enum import auto, Enum
from logging import basicConfig, debug, DEBUG, getLogger, info, INFO
from os import chdir
from os.path import abspath, dirname
from pathlib import Path
from typing import Dict, List, Union

from matplotlib import pyplot

from imu import IMUVector
from matrix import SquareMatrix, Vector, VectorOrientation
from test_data import get_test_data
from ukf import UKF

basicConfig(
    datefmt='%Y-%m-%d:%H:%M:%S',
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=DEBUG,
    )
chdir(dirname(abspath(__file__)))

_CSV_HEADER_PREFIX_DATA = 'DATA_'
_CSV_HEADER_PREFIX_UKF = 'UKF_'
_CSV_OPEN_ENCODING = 'utf-8'
_CSV_READ_MODE = 'r'
_CSV_WRITE_MODE = 'w'
_MAX_MEASUREMENTS_TO_LOAD = 20000
_NUMBER_OF_ITERATIONS_BETWEEN_STATUS_PRINT = 1000
_NUMBER_OF_MEASUREMENTS_TO_SKIP = 10000
_PLOT_COLOR_DATA = 'b'
_PLOT_COLOR_UKF = 'r'
_PREVIOUS_RUN_CSV_FILE_PATH = './previous_run_results.csv'


class _CSVHeader(Enum):
    TIMESTAMP = auto()
    DATA_POSITION_X = auto()
    DATA_POSITION_Y = auto()
    DATA_ACCELERATION_X = auto()
    DATA_ACCELERATION_Y = auto()
    UKF_POSITION_X = auto()
    UKF_POSITION_Y = auto()
    UKF_SPEED_X = auto()
    UKF_SPEED_Y = auto()
    UKF_RMSE = auto()

    def __str__(self) -> str:
        return self.name

    def is_data_field(self) -> bool:
        """ Return True if the field belongs to the test data, and False o.w. """
        return self.name.startswith(_CSV_HEADER_PREFIX_DATA)

    def is_ukf_field(self) -> bool:
        """ Return True if the field belongs to the UKF calculation, and False o.w. """
        return self.name.startswith(_CSV_HEADER_PREFIX_UKF)


class _CSVData(Dict[str, List[float]]):
    """ A wrapper for the CSV data dictionary that support `__getitem__()` using the Enum `_CSVHeader`. """

    def __missing__(self, key: Union[str, _CSVHeader]) -> List[float]:
        if isinstance(key, _CSVHeader):
            return self[key.name]
        raise KeyError(key)


def _load_results_from_previous_run() -> _CSVData:
    """ Load previous results from a UKF run that were stored within a CSV file.

    :note: The function assume that all columns have the same length, if not, an error will be raised from `matplotlib`.
    """
    debug('Loading data from CSV file')
    info('Skip UKF algorithm')
    csv_data = _CSVData({})
    with open(_PREVIOUS_RUN_CSV_FILE_PATH, encoding=_CSV_OPEN_ENCODING, mode=_CSV_READ_MODE) as infile:
        reader = DictReader(infile)
        for row in reader:
            for column_name, column_value in row.items():
                csv_data.setdefault(column_name, []).append(column_value)  # Group the CSV file by columns
    return csv_data


def _plot(csv_data: _CSVData) -> None:
    """ Plot the results into 4 graphs.

    :note: The function contains hardcoded strings for the titles and labels.
    """
    debug('Plotting')
    getLogger().setLevel(INFO)  # Suppress debug information from `matplotlib`
    figure, axis = pyplot.subplots(nrows=2, ncols=2, figsize=(3, 3))
    figure.tight_layout()
    figure.suptitle('Test results')
    axis[0, 0].plot(csv_data[_CSVHeader.DATA_POSITION_X], csv_data[_CSVHeader.DATA_POSITION_Y], _PLOT_COLOR_DATA)
    axis[0, 0].plot(csv_data[_CSVHeader.UKF_POSITION_X], csv_data[_CSVHeader.UKF_POSITION_Y], _PLOT_COLOR_UKF)
    axis[0, 0].set_xlabel('X position [meter]')
    axis[0, 0].set_ylabel('Y position [meter]')
    axis[0, 0].set_title('Vehicle Position')
    axis[0, 1].plot(csv_data[_CSVHeader.TIMESTAMP], csv_data[_CSVHeader.UKF_RMSE], _PLOT_COLOR_UKF)
    axis[0, 1].set_xlabel('Time [second]')
    axis[0, 1].set_ylabel('RMSE')
    axis[0, 1].set_title('Root Mean Squared Error')
    axis[1, 0].plot(csv_data[_CSVHeader.TIMESTAMP], csv_data[_CSVHeader.UKF_SPEED_X], _PLOT_COLOR_UKF)
    axis[1, 0].plot(csv_data[_CSVHeader.TIMESTAMP], csv_data[_CSVHeader.UKF_SPEED_Y], _PLOT_COLOR_UKF)
    axis[1, 0].set_xlabel('Time [second]')
    axis[1, 0].set_ylabel('Speed [m/s]')
    axis[1, 0].set_title('Speed from UKF')
    axis[1, 1].plot(csv_data[_CSVHeader.TIMESTAMP], csv_data[_CSVHeader.DATA_ACCELERATION_X], _PLOT_COLOR_DATA)
    axis[1, 1].plot(csv_data[_CSVHeader.TIMESTAMP], csv_data[_CSVHeader.DATA_ACCELERATION_Y], _PLOT_COLOR_DATA)
    axis[1, 1].set_xlabel('Time [second]')
    axis[1, 1].set_ylabel('Acceleration [m/s^2]')
    axis[1, 1].set_title('Acceleration from data')
    pyplot.show()


def _previous_run_csv_exists() -> bool:
    """ Check if a CSV file from an previous run of the code exists. """
    return Path(_PREVIOUS_RUN_CSV_FILE_PATH).resolve().is_file()


def _run_ukf() -> _CSVData:
    """ Run the Unscented Kalman Filter on a given data. """
    def _load_test_data() -> _CSVData:
        timestamps, relative_x, relative_y, acceleration_x, acceleration_y = get_test_data(
            measurements_to_skip=_NUMBER_OF_MEASUREMENTS_TO_SKIP,
            max_measurements_to_load=_MAX_MEASUREMENTS_TO_LOAD,
            )
        return _CSVData({
            _CSVHeader.TIMESTAMP.name: timestamps,
            _CSVHeader.DATA_POSITION_X.name: relative_x,
            _CSVHeader.DATA_POSITION_Y.name: relative_y,
            _CSVHeader.DATA_ACCELERATION_X.name: acceleration_x,
            _CSVHeader.DATA_ACCELERATION_Y.name: acceleration_y,
            })
    run_data = _load_test_data()
    for header in _CSVHeader:
        if header.is_ukf_field():
            run_data[header] = []
    debug(f'Loaded {len(run_data[_CSVHeader.TIMESTAMP]):,} timestamps from the test file')
    L = len(IMUVector.fields())  # [x, y, v_x, v_y, a_x, a_y]
    ukf = UKF(
        covariance_P=SquareMatrix.diagonal(0.0001, L),  # Initial covariance
        dimension_L=L,
        function_f=IMUVector.function_f,
        function_h=IMUVector.function_h,
        mean_x=Vector.of_size(L),  # Initial state
        process_noise_Q=IMUVector.process_noise_Q(),
        )
    debug('Starting UKF iterations')
    last_timestamp = None
    for data_index, timestamp in enumerate(run_data[_CSVHeader.TIMESTAMP]):
        if data_index % _NUMBER_OF_ITERATIONS_BETWEEN_STATUS_PRINT == 0:
            position_x = f'{run_data[_CSVHeader.DATA_POSITION_X][data_index]:.2f}'
            position_y = f'{run_data[_CSVHeader.DATA_POSITION_Y][data_index]:.2f}'
            debug(f'[{data_index:,}/{len(run_data[_CSVHeader.TIMESTAMP]):,}] Running UKF. ' \
                  f'Timestamp: {timestamp:.1f} sec. Vehicle location: ({position_x},{position_y})')
        time_delta = timestamp - last_timestamp if last_timestamp is not None else 0.0
        last_timestamp = timestamp
        imu_vector = IMUVector(
            x=run_data[_CSVHeader.DATA_POSITION_X][data_index],
            y=run_data[_CSVHeader.DATA_POSITION_Y][data_index],
            a_x=run_data[_CSVHeader.DATA_ACCELERATION_X][data_index],
            a_y=run_data[_CSVHeader.DATA_ACCELERATION_Y][data_index],
            )
        ukf, rmse_value = ukf.step(time_delta=time_delta, imu_vector=imu_vector)
        ukf_state = IMUVector.from_vector(ukf.mean_x)
        run_data[_CSVHeader.UKF_POSITION_X].append(ukf_state.x)
        run_data[_CSVHeader.UKF_POSITION_Y].append(ukf_state.y)
        run_data[_CSVHeader.UKF_SPEED_X].append(ukf_state.v_x)
        run_data[_CSVHeader.UKF_SPEED_Y].append(ukf_state.v_y)
        run_data[_CSVHeader.UKF_RMSE].append(rmse_value)
    return run_data


def _save_data_to_csv(csv_data: _CSVData) -> None:
    """ Save the UKF results to a CSV file. """
    debug('Saving data to a CSV file')
    with open(_PREVIOUS_RUN_CSV_FILE_PATH, encoding=_CSV_OPEN_ENCODING, mode=_CSV_WRITE_MODE) as oufile:
        writer = DictWriter(oufile, csv_data.keys())
        writer.writeheader()
        writer.writerows(csv_data)


if __name__ == '__main__':
    debug('Start')
    if _previous_run_csv_exists():
        _csv_data = _load_results_from_previous_run()
    else:
        _csv_data = _run_ukf()
        _save_data_to_csv(_csv_data)
    _plot(_csv_data)
    debug('Finished')
