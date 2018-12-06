import cvxpy as cvx
import numpy as np


class ControlModel(object):
    """
    Models used to find optimal investment strategies.
    """

    def __init__(self):
        pass

    def run(self, data):
        pass

    def optima(self):
        pass

    def variables(self):
        pass


class NormModel(ControlModel):

    def __init__(self, num_assets, gamma=1.0, regularization=1):
        super(ControlModel, CovarianceModel).__init__(self)
        self.num_assets = num_assets
        self.gamma = gamma
        self.regularization = regularization
        self.x = None
        self.problem = None
        self._optima = None

    def run(self, data):
        mu, sigma = data

        self.x = cvx.Variable(self.num_assets)

        objective = self.x.T*mu - self.gamma*cvx.norm(self.x, self.regularization)

        self.problem = cvx.Problem(cvx.Maximize(objective),
                           [
                               cvx.norm(self.x, 1) <= 1,
                               self.x >= 0
                           ])
        self._optima = self.problem.solve()

    def optima(self):
        return self._optima

    def variables(self):
        return self.x.value.flatten()


class CovarianceModel(ControlModel):

    def __init__(self, num_assets, gamma=1.0):
        super(ControlModel, CovarianceModel).__init__(self)
        self.num_assets = num_assets
        self.gamma = gamma
        self.x = None
        self.problem = None
        self._optima = None

    def run(self, data):
        mu, sigma = data

        self.x = cvx.Variable(self.num_assets)

        objective = self.x.T*mu - self.gamma*cvx.quad_form(self.x, sigma)

        self.problem = cvx.Problem(cvx.Maximize(objective),
                           [
                               cvx.norm(self.x, 1) <= 1,
                               self.x >= 0
                           ])
        self._optima = self.problem.solve()

    def optima(self):
        return self._optima

    def variables(self):
        return self.x.value.flatten()


class ValueAtRiskModel(ControlModel):
    def __init__(self, num_assets, beta=0.0, epsilon=0.05):
        self.num_assets = num_assets
        self.beta = beta
        self.epsilon = epsilon
        self.x = None
        self.problem = None
        self._optima = None

    def run(self, data):
        mu, sigma = data
        self.x = cvx.Variable(self.num_assets)
        objective = self.x.T*mu
        self.problem = cvx.Problem(cvx.Maximize(objective),
            [
                cvx.norm(self.x, 1) <= 1,
                self.x >= 0,
                -self.x.T*mu + np.sqrt((1-self.epsilon)/self.epsilon) * cvx.sqrt(cvx.quad_form(self.x, sigma)) <= self.beta
            ])
        self._optima = self.problem.solve()

    def optima(self):
        return self._optima

    def variables(self):
        return self.x.value.flatten()


if __name__ == "__main__":
    from data_models import GaussianNoise
    from prediction_models import UnbiasEstimator

    num_samples = 1000
    num_assets = 3

    mu_truth = np.ones(num_assets)
    sigma_truth = np.diag([0.5, 0.3, 0.2])

    sampler = GaussianNoise()
    data = np.zeros(shape=(num_samples, num_assets))

    for i in range(num_samples):
        data[i] = sampler.sample((mu_truth, sigma_truth))

    sample_mean, sample_covar = UnbiasEstimator().predict(data)

    for i in range(num_assets):
        print(sample_mean[i], sample_covar[i])

    cov_model = CovarianceModel(num_assets=num_assets)
    cov_model.run(data=(sample_mean, sample_covar), gamma=1.0)

    print(cov_model.variables())
    print(cov_model.optima())
