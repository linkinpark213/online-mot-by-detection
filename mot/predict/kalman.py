"""
Implementation code for Kalman filter is borrowed from https://github.com/nwojke/deep_sort with minor modifications.

"""
import scipy
import numpy as np
from mot.predict import Predictor


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    ndim, dt = 4, 1.
    _motion_mat = np.eye(2 * ndim, 2 * ndim)
    for i in range(ndim):
        _motion_mat[i, ndim + i] = dt
    _update_mat = np.eye(ndim, 2 * ndim)
    # Motion and observation uncertainty are chosen relative to the current
    # state estimate. These weights control the amount of uncertainty in
    # the model. This is a bit hacky.
    _std_weight_position = 1. / 20
    _std_weight_velocity = 1. / 160

    def __init__(self, measurement):
        self.mean, self.covariance = self.initiate(measurement)

    @classmethod
    def initiate(self, measurement):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    @classmethod
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    @classmethod
    def project(self, mean, covariance):
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    @classmethod
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    @classmethod
    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha


class KalmanPredictor(Predictor):
    def __init__(self, box_type='xyxy', predict_type='xyah'):
        super(KalmanPredictor).__init__()
        self.box_type = box_type
        self.predict_type = predict_type

    def __call__(self, tracklet):
        return self.predict(tracklet)

    def convert(self, box, in_type, out_type):
        assert in_type in ['xyxy', 'xywh', 'xyah'] and out_type in ['xyxy', 'xywh',
                                                                    'xyah'], "Unknown box representation"
        if in_type == 'xyxy':
            if out_type == 'xyah':
                return self.xyxy2xyah(box)
            elif out_type == 'xywh':
                return self.xyxy2xywh(box)
        elif in_type == 'xyah':
            if out_type == 'xyxy':
                return self.xyah2xyxy(box)
            elif out_type == 'xywh':
                return self.xyah2xywh(box)
        elif in_type == 'xywh':
            if out_type == 'xyxy':
                return self.xywh2xyxy(box)
            elif out_type == 'xyah':
                return self.xywh2xyah(box)

    def xyxy2xywh(self, box):
        measurement = box.copy()[:4]
        measurement[0] = (box[0] + box[2]) / 2
        measurement[1] = (box[1] + box[3]) / 2
        measurement[2] = box[2] - box[0]
        measurement[3] = box[3] - box[1]
        return measurement

    def xyxy2xyah(self, box):
        measurement = self.xyxy2xywh(box)
        measurement[2] = measurement[2] / measurement[3]
        return measurement

    def xyah2xywh(self, box):
        measurement = box.copy()[:4]
        measurement[2] = measurement[2] * measurement[3]
        return measurement

    def xyah2xyxy(self, box):
        measurement = self.xyah2xywh(box)
        return self.xywh2xyxy(measurement)

    def xywh2xyxy(self, box):
        measurement = box.copy()[:4]
        measurement[0] = measurement[0] - measurement[2] / 2
        measurement[1] = measurement[1] - measurement[3] / 2
        measurement[2] = measurement[0] + measurement[2]
        measurement[3] = measurement[1] + measurement[3]
        return measurement

    def xywh2xyah(self, box):
        measurement = box.copy()[:4]
        measurement[2] = measurement[2] / measurement[3]
        return measurement

    def initiate(self, tracklet):
        measurement = self.convert(tracklet.last_box, self.box_type, self.predict_type)
        tracklet.mean, tracklet.covariance = KalmanFilter.initiate(measurement)

    def update(self, tracklet):
        measurement = self.convert(tracklet.last_box, self.box_type, self.predict_type)
        tracklet.mean, tracklet.covariance = KalmanFilter.update(tracklet.mean, tracklet.covariance, measurement)

    def predict(self, tracklet):
        tracklet.mean, tracklet.covariance = KalmanFilter.predict(tracklet.mean, tracklet.covariance)
        return self.convert(tracklet.mean, self.predict_type, self.box_type)
