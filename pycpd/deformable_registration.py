from builtins import super
import numpy as np
from .expectation_maximization_registration import expectation_maximization_registration

def make_kernel(Y, beta):
    (M, D) = Y.shape
    XX = np.reshape(Y, (1, M, D))
    YY = np.reshape(Y, (M, 1, D))
    XX = np.tile(XX, (M, 1, 1))
    YY = np.tile(YY, (1, M, 1))
    diff = XX-YY
    diff = np.multiply(diff, diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta))

class deformable_registration(expectation_maximization_registration):
    def __init__(self, alpha=2, beta=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha         = 2 if alpha is None else alpha
        self.beta          = 2 if alpha is None else beta
        self.W             = np.zeros((self.M, self.D))
        self.G             = make_kernel(self.Y, self.beta)

    def update_transform(self):
        A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
        B = np.dot(self.P, self.X) - np.dot(np.diag(self.P1), self.Y)
        self.W = np.linalg.solve(A, B)

    def transform_point_cloud(self, Y=None):
        if Y is None:
            self.TY = self.Y + np.dot(self.G, self.W)
            return
        else:
            return Y + np.dot(self.G, self.W)

    def update_variance(self):
        qprev = self.sigma2

        xPx      = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.X, self.X), axis=1))
        yPy      = np.dot(np.transpose(self.P1),  np.sum(np.multiply(self.Y, self.Y), axis=1))
        trPXY    = np.sum(np.multiply(self.Y, np.dot(self.P, self.X)))

        X_dPt1_X = np.matmul(np.transpose(self.X), np.matmul(np.diag(self.Pt1), self.X))
        P_XT_TY = np.matmul(np.transpose(np.matmul(self.P, self.X)), self.TY)
        TYT_dP1_TY = np.matmul(np.transpose(self.TY), np.matmul(np.diag(self.P1), self.TY))
        self.sigma2 = (np.trace(X_dPt1_X) - 2 * np.trace(P_XT_TY) +
                       np.trace(TYT_dP1_TY)) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10
        self.err = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        return self.G, self.W
