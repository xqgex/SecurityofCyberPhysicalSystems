""" Implementation of Unscented Kalman Filter (UKF).

Note:
The Docstrings equations are according to Wan et al., `The unscented Kalman filter for nonlinear estimation`.

> Algorithm 3.1: Unscented Kalman Filter (UKF) equations

Some publications may have deviations from these symbols, e.g.:

* Replacing sigma points `X^a` and `Y` with `Y` and `Z`.
* Replacing process/measurement noise `P_v` and `P_n` with `Q` and `R`.

Although the logic remains the same, please be aware of such deviations...

The implementation is based on the following papers:

1. Simon J. Julier and Jeffrey K. Uhlmann. 1997. New extension of the Kalman filter
   to nonlinear systems. In Signal Processing, Sensor Fusion, and Target Recognition
   VI, Ivan Kadar (Ed.), Vol. 3068. International Society for Optics and Photonics,
   SPIE, 182–193. https://doi.org/10.1117/12.280797
2. E.A. Wan and R. Van Der Merwe. 2000. The unscented Kalman filter for nonlinear
   estimation. In Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing,
   Communications, and Control Symposium (Cat. No.00EX373). 153–158.
   https://doi.org/10.1109/ASSPCC.2000.882463

More information about the algorithm and the papers that have been used can be found at:
https://github.com/xqgex/SecurityofCyberPhysicalSystems/tree/main/papers

This code was proudly written using Gedit.
Released under the following license:
https://github.com/xqgex/SecurityofCyberPhysicalSystems/blob/main/LICENSE
"""

from functools import cached_property
from math import sqrt as math_sqrt
from typing import Any, Callable, Dict, Tuple

import attr
from attr.validators import ge, gt, instance_of

from imu import IMUVector
from matrix import Matrix, SquareMatrix, Vector, VectorOrientation

_UKF_ALPHA = 0.001  # 0 < _UKF_ALPHA
_UKF_BETA = 2  # 0 <= _UKF_BETA
_UKF_KAPPA = 0  # 0 <= _UKF_KAPPA


@attr.s(frozen=True, kw_only=True, slots=True)
class UKF:
    """ Implementation of Unscented Kalman Filter (UKF). """

    __dict__: Dict[str, Any] = attr.ib(
        eq=False,
        factory=dict,
        init=False,
        metadata={'class': 'UKF'},
        repr=False,
        ) # To support cached properties <https://github.com/python-attrs/attrs/issues/164#issuecomment-1233241377>.
    _alpha: float = attr.ib(
        default=_UKF_ALPHA,
        validator=[instance_of(float), gt(0.0)],
        )  # Determines the spread of the sigma points around `x`, usually set to a small positive value, e.g. `1e-3`.
    _beta: int = attr.ib(
        default=_UKF_BETA,
        validator=[instance_of(int), ge(0)],
        )  # Used to incorporate prior knowledge of the distribution of `x`, for Gaussian distributions `2` is optimal.
    _kappa: int = attr.ib(
        default=_UKF_KAPPA,
        validator=[instance_of(int), ge(0)],
        )  # Excess kurtosis, a secondary scaling parameter, `0` for Normal Distribution (`3` for Laplace Distribution).
    covariance_P: SquareMatrix = attr.ib(
        validator=instance_of(SquareMatrix),
        )  # Initial covariance matrix of size `L` by `L`.
    dimension_L: int = attr.ib(
        validator=instance_of(int),
        )  # Number of states in `x` (the dimension of the augmented state).
    function_f = attr.ib(
        validator=instance_of(Callable),
        )  # Function :math:`F()` used for prediction based on time delta and the process noise.
    function_h = attr.ib(
        validator=instance_of(Callable),
        )  # Function :math:`H()` used for prediction based on the measurements noise.
    mean_x: Vector = attr.ib(
        validator=instance_of(Vector),
        )  # Mean `x` is a column vector of size `L` represents the initial values for the states.
    process_noise_Q: SquareMatrix = attr.ib(
        validator=instance_of(SquareMatrix),
        )  # The process noise Q is an `L` by `L` matrix represents the (constant) uncertainty.

    def __attrs_post_init__(self) -> None:
        """ Post initialization validations. """
        if len(self.covariance_P) != self.dimension_L:
            raise ValueError(
                'The size of the covariance matrix `P` does not match the dimension `L`.\nExpected ' \
                f'{self.dimension_L}X{self.dimension_L} matrix, got {len(self.covariance_P)}X{len(self.covariance_P)}')
        if len(self.mean_x) != self.dimension_L:
            raise ValueError(
                'The size of the mean vector `x` does not match the dimension `L`.\n' \
                f'Expected a vector of size {self.dimension_L}, got {len(self.mean_x)}')
        if len(self.process_noise_Q) != self.dimension_L:
            raise ValueError(
                'The size of the noise matrix `Q` does not match the dimension `L`.\n' \
                f'Expected {self.dimension_L}X{self.dimension_L} matrix\n' \
                f'Got {len(self.process_noise_Q)}X{len(self.process_noise_Q)}')

    def __str__(self) -> str:
        """ A utility to represents the UKF instance. """
        return '\n'.join([f'{a.name}: {self.__getattribute__(a.name)}' for a in self.__attrs_attrs__ if a.repr])

    def _predict_covariance_P(
            self,
            sigma_points_predict_X: Tuple[Vector, ...],
            mean_predict_x_k: Vector
            ) -> SquareMatrix:
        r""" The predicted covariance :math:`P_k^{-}`.

        :equation:

        .. math::
           P_k^{-} = \sum_{i=0}^{2L} W_i^{(c)} [\mathcal{X}_{i,k|k - 1}^x - \hat{x}_k^{-}]
                     [\mathcal{X}_{i,k|k - 1}^x - \hat{x}_k^{-}]^T

        :param Tuple[Vector, ...] sigma_points_predict_X: The predicted sigma points X, 2L+1 vectors of size L.
        :param Vector mean_predict_x_k: The predicted mean, a vector of size L.
        :returns: The predicted covariance, a square matrix of size L by L.
        :rtype: SquareMatrix
        """
        covariance_predict_P = SquareMatrix.of_size(self.dimension_L)
        for index in range(self._sigmas_count):
            # The variable `self._weights_covariance_w_c` is a vector of size 2L+1
            x_diff: Vector = sigma_points_predict_X[index] - mean_predict_x_k
            covariance_predict_P += self._weights_covariance_w_c[index] * x_diff * x_diff.transpose
        return covariance_predict_P

    def _predict_cross_correlations(
            self,
            measurement_indices: Tuple[int, ...],
            sigma_points_predict_X: Tuple[Vector, ...],
            sigma_points_predict_Y: Tuple[Vector, ...],
            mean_predict_x_k: Vector,
            observation_predict_y_k: Vector
            ) -> Tuple[SquareMatrix, SquareMatrix]:
        r""" The predicted innovation covariance :math:`P_{y_k y_k}` and cross correlation matrix :math:`P_{x_k y_k}`.

        :equation:

        .. math::
           \begin{align*}
             P_{y_k y_k} &= \sum_{i=0}^{2L} W_i^{(c)} [\mathcal{Y}_{i,k|k - 1} - \hat{y}_k^{-}]
                            [\mathcal{Y}_{i,k|k - 1} - \hat{y}_k^{-}]^T \\
             P_{x_k y_k} &= \sum_{i=0}^{2L} W_i^{(c)} [\mathcal{X}_{i,k|k - 1}^{x} - \hat{x}_k^{-}]
                            [\mathcal{Y}_{i,k|k - 1} - \hat{y}_k^{-}]^T
           \end{align*}

        :param Tuple[int, ...] measurement_indices: The indices of the measurements in the IMU vector.
        :param Tuple[Vector, ...] sigma_points_predict_X: The predicted sigma points X, 2L+1 vectors of size L.
        :param Tuple[Vector, ...] sigma_points_predict_Y: The predicted sigma points Y, 2L+1 vectors of size
                                                          `len(imu_vector)`.
        :param Vector mean_predict_x_k: The predicted mean, a vector of size L.
        :param Vector observation_predict_y_k: The predicted observation, a column vector of size `len(imu_vector)`.
        :returns: The predicted innovation covariance is a square matrix of size `len(imu_vector)` by `len(imu_vector)`.
                  The cross correlation is a matrix of size `L` by `len(imu_vector)`.
        :rtype: Tuple[SquareMatrix, SquareMatrix]
        """
        innovation_covariance_predict_P_yy = SquareMatrix.of_size(len(measurement_indices))
        cross_correlation_predict_P_xy = Matrix.of_size(self.dimension_L, len(measurement_indices))
        for index in range(self._sigmas_count):
            # The variable `self._weights_covariance_w_c` is a vector of size 2L+1
            # Note, `x_diff` is a vector of size `L` and `y_diff` is a vector of size `len(imu_vector)`
            x_diff: Vector = sigma_points_predict_X[index] - mean_predict_x_k
            y_diff: Vector = sigma_points_predict_Y[index] - observation_predict_y_k
            innovation_covariance_predict_P_yy += self._weights_covariance_w_c[index] * y_diff * y_diff.transpose
            cross_correlation_predict_P_xy += self._weights_covariance_w_c[index] * x_diff * y_diff.transpose
        return innovation_covariance_predict_P_yy, cross_correlation_predict_P_xy

    def _predict_mean_x_k(self, sigma_points_predict_X: Tuple[Vector, ...]) -> Vector:
        r""" The predicted mean :math:`\hat{x}_k^{-}`.

        :equation:

        .. math::
           \hat{x}_k^{-} = \sum_{i=0}^{2L} W_i^{(m)} \mathcal{X}_{i,k|k - 1}^x

        :param Tuple[Vector, ...] sigma_points_predict_X: The predicted sigma points X, 2L+1 vectors of size L.
        :returns: The predicted mean, a vector of size L.
        :rtype: Vector
        """
        mean_predict_x_k = Vector.of_size(self.dimension_L)
        for index, sigma_point in enumerate(sigma_points_predict_X):
            # The variable `self._weights_mean_w_m` is a vector of size 2L+1
            mean_predict_x_k += self._weights_mean_w_m[index] * sigma_point
        return mean_predict_x_k

    def _predict_observation_y_k(self, measurements_count: int, sigma_points_predict_Y: Tuple[Vector, ...]) -> Vector:
        r""" The predicted observation :math:`\hat{y}_k^{-}`.

        :equation:

        .. math::
           \hat{y}_k^{-} = \sum_{i=0}^{2L} W_i^{(m)} \mathcal{Y}_{i,k|k - 1}

        :param int measurements_count: The size of the measurements vector.
        :param Tuple[Vector, ...] sigma_points_predict_Y: The predicted sigma points Y, 2L+1 vectors of size L.
        :returns: The predicted observation, a column vector of size `len(imu_vector)`.
        :rtype: Vector
        """
        observation_predict_y_k = Vector.of_size(measurements_count)
        for index, sigma_point in enumerate(sigma_points_predict_Y):
            # The variable `self._weights_mean_w_m` is a vector of size 2L+1
            # The variable `sigma_point` is a vector of size `len(imu_vector)`
            observation_predict_y_k += self._weights_mean_w_m[index] * sigma_point
        return observation_predict_y_k

    def _predict_sigma_points_X(self, time_delta: float) -> Tuple[Vector, ...]:
        r""" Predict sigma points :math:`\mathcal{X}_{k|k - 1}^x` based on the process noise.

        > The transformed set is given by instantiating each point through the process model.

        :equation:

        .. math::
           \textbf{Predict sigma points X} \\
           \begin{matrix}
             \mathcal{X}_{i,k|k - 1}^x = F(\mathcal{X}_{i,k - 1}^a, \mathcal{X}^v) & i = 0 \ldots 2L
           \end{matrix}

        :param float time_delta: The time delta since the previous step.
        :returns: The predicted sigma points, 2L+1 column vectors with L elements each.
        :rtype: Tuple[Vector, ...]
        """
        def _calculate_point(sigma_point_index: int) -> Vector:
            return self.function_f(self._instance_sigma_points[sigma_point_index], self.process_noise_Q, time_delta)
        return tuple(_calculate_point(i) for i in range(self._sigmas_count))

    def _predict_sigma_points_Y(self, imu_vector: IMUVector) -> Tuple[Vector, ...]:
        r""" Predict sigma points :math:`\mathcal{Y}_{k|k - 1}` based on the measurement noise.

        > Instantiate each of the prediction points through the observation model.

        :equation:

        .. math::
           \textbf{Predict sigma points Y} \\
           \begin{matrix}
             \mathcal{Y}_{i,k|k - 1} = H(\mathcal{X}_{i,k|k - 1}^x, \mathcal{X}^n) & i = 0 \ldots 2L
           \end{matrix}

        :param IMUVector imu_vector: The IMU vector with the measurements information.
        :returns: The predicted sigma points, 2L+1 column vectors with `len(imu_vector)` elements each.
        :rtype: Tuple[Vector, ...]
        """
        def _calculate_point(sigma_point_index: int) -> Vector:
            vector = self._instance_sigma_points[sigma_point_index].filter_with_indices(imu_vector.indices)
            return self.function_h(vector, imu_vector.measurement_noise_R)
        return tuple(_calculate_point(i) for i in range(self._sigmas_count))

    def predict(self, time_delta: float) -> Tuple[Tuple[Vector, ...], Vector, SquareMatrix]:
        r""" Predict a mean state and covariance matrix from the sigma points.

        > Predict the new state of the system :math:`\hat{x}_k^{-}` and its associated covariance :math:`P_k^{-}`.
        > This prediction must take account of the effects of process noise.

        :note: The location where the noise is added cause an unsolvable bug. the process noise was moved from
               `IMUVector.function_f()` to here `UKF().predict()`, both places are marked with a comment
               `# XXX XXX XXX`.

        :equation:

        .. math::
           \textbf{UKF predict} \\
           \begin{align*}
             \mathcal{X}_{k|k - 1}^x &= F(\mathcal{X}_{k - 1}^a, \mathcal{X}^v) \\
             \hat{x}_k^{-} &= \sum_{i=0}^{2L} W_i^{(m)} \mathcal{X}_{i,k|k - 1}^x \\
             P_k^{-} &= \sum_{i=0}^{2L} W_i^{(c)} [\mathcal{X}_{i,k|k - 1}^x - \hat{x}_k^{-}]
                        [\mathcal{X}_{i,k|k - 1}^x - \hat{x}_k^{-}]^T
           \end{align*}

        :param float time_delta: The time delta since the previous step.
        :returns: The predicted sigma points, mean and covariance.
        :rtype: Tuple[Tuple[Vector, ...], Vector, SquareMatrix]
        """
        sigma_points_predict_X = self._predict_sigma_points_X(time_delta)
        mean_predict_x_k = self._predict_mean_x_k(sigma_points_predict_X)
        covariance_predict_P = self._predict_covariance_P(sigma_points_predict_X, mean_predict_x_k)
        covariance_predict_P += time_delta * self.process_noise_Q  # XXX XXX XXX
        return sigma_points_predict_X, mean_predict_x_k, covariance_predict_P

    def step(self, time_delta: float, imu_vector: IMUVector) -> Tuple['UKF', float]:
        """ Perform consecutive prediction (based on a given time delta) and an update (using a given IMU vector).

        :param float time_delta: The time delta since the previous step.
        :param IMUVector imu_vector: An IMU vector with the new measurements.
        :returns: A UKF instance ready for the next iteration, and the prediction RMSE.
        :rtype: Tuple[UKF, float]
        """
        sigma_points_predict_X, mean_predict_x_k, covariance_predict_P = self.predict(time_delta)
        return self.update(sigma_points_predict_X, mean_predict_x_k, covariance_predict_P, imu_vector)

    def update(
            self,
            sigma_points_predict_X: Tuple[Vector, ...],
            mean_predict_x_k: Vector,
            covariance_predict_P: SquareMatrix,
            imu_vector: IMUVector
            ) -> Tuple['UKF', float]:
        r""" Update the state mean and covariance matrix with the measurements.

        > Predict the expected observation :math:`\hat{y}_k^{-}` and the innovation covariance :math:`P_{y_k y_k}`.
        > This prediction should include the effects of observation noise.
        > Finally, predict the cross-correlation matrix :math:`P_{x_k y_k}`.

        :note: The location where the noise is added cause an unsolvable bug. the measurements noise was moved from
               `IMUVector.function_h()` to here `UKF().update()`, both places are marked with a comment `# XXX XXX XXX`.

        :equation:

        .. math::
           \textbf{UKF update} \\
           \begin{align*}
             \mathcal{Y}_{k|k - 1} &= H(\mathcal{X}_{k|k - 1}^x, \mathcal{X}^n) \\
             \hat{y}_k^{-} &= \sum_{i=0}^{2L} W_i^{(m)} \mathcal{Y}_{i,k|k - 1} \\
             P_{y_k y_k} &= \sum_{i=0}^{2L} W_i^{(c)} [\mathcal{Y}_{i,k|k - 1} - \hat{y}_k^{-}]
                            [\mathcal{Y}_{i,k|k - 1} - \hat{y}_k^{-}]^T \\
             P_{x_k y_k} &= \sum_{i=0}^{2L} W_i^{(c)} [\mathcal{X}_{i,k|k - 1}^{x} - \hat{x}_k^{-}]
                            [\mathcal{Y}_{i,k|k - 1} - \hat{y}_k^{-}]^T \\
             \mathcal{K} &= P_{x_k y_k} P_{y_k y_k}^{-1} \\
             \hat{x}_k &= \hat{x}_k^{-} + \mathcal{K} (y_k - \hat{y}_k^{-}) \\
             P_k &= P_k^{-} - \mathcal{K} P_{y_k y_k} \mathcal{K}^T
           \end{align*}

        For evaluation purposes, the RMSE is calculated as follow:

        :equation:

        .. math::
           RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y_{i,k} - \hat{y}_{i,k}^{-})^2}

        :param Tuple[Vector, ...] sigma_points_predict_X: The predicted sigma points X, 2L+1 vectors of size L.
        :param Vector mean_predict_x_k: The predicted mean, a vector of size L.
        :param SquareMatrix covariance_predict_P: SquareMatrix.
        :param IMUVector imu_vector: An IMU vector with the new measurements.
        :returns: A UKF instance ready for the next iteration, and the prediction RMSE.
        :rtype: Tuple[UKF, float]
        """
        sigma_points_predict_Y = self._predict_sigma_points_Y(imu_vector)
        observation_predict_y_k = self._predict_observation_y_k(len(imu_vector), sigma_points_predict_Y)
        innovation_covariance_predict_P_yy, cross_correlation_predict_P_xy = self._predict_cross_correlations(
                                                                                 imu_vector.indices,
                                                                                 sigma_points_predict_X,
                                                                                 sigma_points_predict_Y,
                                                                                 mean_predict_x_k,
                                                                                 observation_predict_y_k)
        innovation_covariance_predict_P_yy += imu_vector.measurement_noise_R  # XXX XXX XXX
        kalman_gain: Matrix = cross_correlation_predict_P_xy * innovation_covariance_predict_P_yy.inverse()
        y_diff = (imu_vector - observation_predict_y_k).to_minimal_vector()
        mean_x = mean_predict_x_k + (kalman_gain * y_diff).to_vector()
        covariance_P = covariance_predict_P - (kalman_gain * innovation_covariance_predict_P_yy * kalman_gain.transpose)
        rmse = math_sqrt(sum(pow(v, 2) for v in y_diff) / len(y_diff))
        return attr.evolve(self, mean_x=mean_x, covariance_P=covariance_P), rmse

    @mean_x.validator
    def _mean_x_validator(self, _, value: Vector):
        if value.orientation == VectorOrientation.HORIZONTAL:
            raise ValueError(f'The mean vector orientation expected to be `vertical`, got `horizontal`.\n`x`: {value}')

    @cached_property
    def _lambda(self) -> float:
        r""" A scaling parameter :math:`\lambda`.

        :note: This is a cached property of the UKF instance.
        :complexity: :math:`O(1)`
        :equation:

        .. math::
           \lambda = \alpha^2 (L + \kappa) - L
        """
        return pow(self._alpha, 2) * (self.dimension_L + self._kappa) - self.dimension_L

    @cached_property
    def _instance_sigma_points(self) -> Tuple[Vector, ...]:
        r""" The sigma points (state distribution) matrix :math:`\mathcal{X}_{k - 1}^a`.

        :note: This is a cached property of the UKF instance.
        :complexity: :math:`O(L^3 + \frac{11}{3} L^2)`
        :equation:

        .. math::
           \textbf{Calculate sigma points} \\
           \mathcal{X}_{i,k - 1}^a = \begin{cases}
             \hat{x}_{k - 1}^a & i = 0 \\
             \hat{x}_{k - 1}^a + (\sqrt{(L + \lambda) P_{k - 1}^a})_i & i = 1 \ldots L \\
             \hat{x}_{k - 1}^a - (\sqrt{(L + \lambda) P_{k - 1}^a})_{i - L} & i = L + 1 \ldots 2L
           \end{cases}
        .. math::
           \begin{equation}
             \mathcal{X}_{k - 1}^a =
             \begin{bmatrix}
               \hat{x}_{k - 1}^a &
               \hat{x}_{k - 1}^a + \sqrt{(L + \lambda) P_{k - 1}^a} &
               \hat{x}_{k - 1}^a - \sqrt{(L + \lambda) P_{k - 1}^a} \\
               \star &
               \star \star \star \star \star \star &
               \star \star \star \star \star \star \\
             \end{bmatrix}^T
           \end{equation}

        >>> ukf = UKF.empty_instance(size=5)
        >>> len(ukf._instance_sigma_points)  # == 2L+1
        11
        >>> {len(sigma_point) for sigma_point in ukf._instance_sigma_points}  # == L
        {5}

        :returns: The sigma points, 2L+1 column vectors with L elements each.
        :rtype: Tuple[Vector, ...]
        """
        matrix_square_root = ((self.dimension_L + self._lambda) * self.covariance_P).square_root()
        return tuple(self.mean_x
                     + _utility_sign(self.dimension_L, index)
                     * matrix_square_root.column(_utility_index(self.dimension_L, index))
                     for index in range(self._sigmas_count))

    @cached_property
    def _sigmas_count(self) -> int:
        r""" The sigma point count.

        :note: This is a cached property of the UKF instance.
        :complexity: :math:`O(1)`

        :returns: How many sigma point the system have (2L+1).
        :rtype: int
        """
        return 2 * self.dimension_L + 1

    @cached_property
    def _weights_covariance_w_c(self) -> Vector:
        r""" The corresponding covariance weights for the matrix of the (2L+1) sigma vectors :math:`W_i^{(c)}`.

        :note: This is a cached property of the UKF instance.
        :complexity: :math:`O(1)`
        :equation:

        .. math::
           W_i^{(c)} = \begin{cases}
             \frac{\lambda}{L + \lambda} + (1 - \alpha^2 + \beta) & i = 0 \\
             \frac{1}{2(L + \lambda)} & i = 1 \ldots 2L
           \end{cases}

        >>> ukf = UKF.empty_instance(size=5)
        >>> len(ukf._weights_covariance_w_c) == ukf._sigmas_count
        True
        >>> len(ukf._weights_covariance_w_c)
        11

        :returns: The weighted sample covariance of the posterior sigma points, a vector of size 2L+1.
        :rtype: Vector
        """
        return Vector(tuple(((self._lambda if index == 0 else 0.5) / (self.dimension_L + self._lambda))
                            + _utility_w_c_i(index, self._alpha, self._beta)
                            for index in range(self._sigmas_count)))

    @cached_property
    def _weights_mean_w_m(self) -> Vector:
        r""" The corresponding mean weights for the matrix of the (2L+1) sigma vectors :math:`W_i^{(m)}`.

        :note: This is a cached property of the UKF instance.
        :complexity: :math:`O(1)`
        :equation:

        .. math::
           W_i^{(m)} = \begin{cases}
             \frac{\lambda}{L + \lambda} & i = 0 \\
             \frac{1}{2(L + \lambda)} & i = 1 \ldots 2L
           \end{cases}

        >>> ukf = UKF.empty_instance(size=5)
        >>> len(ukf._weights_mean_w_m) == ukf._sigmas_count
        True
        >>> len(ukf._weights_mean_w_m)
        11

        :returns: The weighted sample mean of the posterior sigma points, a vector of size 2L+1.
        :rtype: Vector
        """
        return Vector(tuple((self._lambda if index == 0 else 0.5) / (self.dimension_L + self._lambda)
                            for index in range(self._sigmas_count)))

    @classmethod
    def empty_instance(cls, size: int) -> 'UKF':
        """ Return a UKF instance initialized with zeros. """
        return cls(
            covariance_P=SquareMatrix.diagonal(1.0, size),
            dimension_L=size,
            function_f=lambda v, *_: v,
            function_h=lambda v, *_: v,
            mean_x=Vector.of_size(size),
            process_noise_Q=SquareMatrix.of_size(size),
            )


def _utility_index(length: int, index: int) -> int:
    """ Utility function for :func:`UKF._instance_sigma_points`.

    >>> l = 5
    >>> [_utility_index(l, i) for i in range(2 * l + 1)]
    [0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    """
    return index - length - 1 if length < index <= 2 * length else index - 1 if 0 < index <= length else 0


def _utility_sign(length: int, index: int) -> int:
    """ Utility function for :func:`UKF._instance_sigma_points`.

    >>> l = 5
    >>> [_utility_sign(l, i) for i in range(2 * l + 1)]
    [0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
    """
    return -1 if length < index <= 2 * length else 1 if 0 < index <= length else 0


def _utility_w_c_i(index: int, alpha: float, beta: int) -> float:
    """ Utility function for :func:`UKF._weights_covariance_w_c`.

    >>> [_utility_w_c_i(i, _UKF_ALPHA, _UKF_BETA) for i in range(5)]
    [2.999999, 0.0, 0.0, 0.0, 0.0]
    """
    return 1 - pow(alpha, 2) + beta if index == 0 else 0.0


# In order for the @cached_property Doctests to be discovered.
# <https://stackoverflow.com/a/72500890/4678126>
# pylint: disable=protected-access
__test__= {
    'UKF._lambda': UKF._lambda,
    'UKF._instance_sigma_points': UKF._instance_sigma_points,
    'UKF._sigmas_count': UKF._sigmas_count,
    'UKF._weights_covariance_w_c': UKF._weights_covariance_w_c,
    'UKF._weights_mean_w_m': UKF._weights_mean_w_m,
    }
# pylint: enable=protected-access
