from builtins import super
import numpy as np
from .emregistration import EMRegistration
from .utility import is_positive_semi_definite


class AffineRegistration(EMRegistration):
    """
    Affine registration.

    Attributes
    ----------
    B: numpy array (semi-positive definite)
        DxD affine transformation matrix.

    t: numpy array
        1xD initial translation vector.

    YPY: float
        Denominator value used to update the scale factor.
        Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    X_hat: numpy array
        Centered target point cloud.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf

    """

    def __init__(self, B=None, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if B is not None and (B.ndim is not 2 or B.shape[0] is not self.D or B.shape[1] is not self.D or not is_positive_semi_definite(B)):
            raise ValueError(
                'The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.'.format(self.D, self.D, B))

        if t is not None and (t.ndim is not 2 or t.shape[0] is not 1 or t.shape[1] is not self.D):
            raise ValueError(
                'The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.'.format(self.D, t))
        self.B = np.eye(self.D) if B is None else B
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """

        # source and target point cloud means
        muX = np.divide(np.sum(self.PX, axis=0), self.Np)
        muY = np.divide(
            np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.X_hat = self.X - np.tile(muX, (self.N, 1))
        Y_hat = self.Y - np.tile(muY, (self.M, 1))

        self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)

        self.YPY = np.dot(np.transpose(Y_hat), np.diag(self.P1))
        self.YPY = np.dot(self.YPY, Y_hat)

        # Calculate the new estimate of affine parameters using update rules for (B, t)
        # as defined in Fig. 3 of https://arxiv.org/pdf/0905.2635.pdf.
        self.B = np.linalg.solve(np.transpose(self.YPY), np.transpose(self.A))
        self.t = np.transpose(
            muX) - np.dot(np.transpose(self.B), np.transpose(muY))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the affine transformation.

        """
        if Y is None:
            self.TY = np.dot(self.Y, self.B) + np.tile(self.t, (self.M, 1))
            return
        else:
            return np.dot(Y, self.B) + np.tile(self.t, (Y.shape[0], 1))

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the affine transformation.
        See the update rule for sigma2 in Fig. 3 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.q

        trAB = np.trace(np.dot(self.A, self.B))
        xPx = np.dot(np.transpose(self.Pt1), np.sum(
            np.multiply(self.X_hat, self.X_hat), axis=1))
        trBYPYP = np.trace(np.dot(np.dot(self.B, self.YPY), self.B))
        self.q = (xPx - 2 * trAB + trBYPYP) / (2 * self.sigma2) + \
            self.D * self.Np/2 * np.log(self.sigma2)
        self.diff = np.abs(self.q - qprev)

        self.sigma2 = (xPx - trAB) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the affine transformation parameters.

        """
        return self.B, self.t
