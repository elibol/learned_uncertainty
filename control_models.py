import cvxpy as cvx
import numpy as np
from scipy.linalg import sqrtm

import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
plt.rc('figure', figsize=[10, 6])

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

    def get_input(self, L, past_data, ar_projections, ar_errors, bank_rate=1.0):
        pass

    def apply_model_results(self, true_x):
        pass


class NormModel(ControlModel):

    def __init__(self, num_assets, gamma=1.0, norm=1, nu=0.01):
        super(ControlModel, CovarianceModel).__init__(self)
        self.num_assets = num_assets
        self.gamma = gamma
        self.norm = norm
        self.nu = nu
        self.x = None
        self.x0 = None
        self.problem = None
        self._optima = None

    def run(self, data):
        x0, mu, sigma = data
        self.x0 = x0

        self.x = cvx.Variable(self.num_assets+1)

        objective = self.x.T*mu - self.gamma*cvx.norm(self.x, self.norm)

        self.problem = cvx.Problem(cvx.Maximize(objective),
                           [
                               cvx.norm(self.x, 1) <= 1 - self.nu * cvx.norm(self.x - x0, 1),
                               self.x >= 0
                           ])
        self._optima = self.problem.solve()

    def optima(self):
        return self._optima

    def variables(self):
        return self.x.value.flatten()

    def get_input(self, L, past_data, ar_projections, ar_errors, bank_rate=1.0):
        proj = ar_projections.T[:,0]
        proj = np.concatenate((proj, np.array([bank_rate])))
        if len(ar_errors.shape) == 3:
            # janky way to tell what kind of error we have available.
            return proj, ar_errors[0]
        else:
            return proj, ar_errors.reshape(ar_errors.shape[0], 1)[:,0]

    def apply_model_results(self, true_x):
        new_x = self.variables()
        updated_x = true_x.copy()
        # modify updated_x with sell/buy changes.
        sell, buy = np.maximum(0, -(new_x - true_x)), np.maximum(0, (new_x - true_x))
        updated_x = updated_x - sell + buy
        return updated_x


class CovarianceModel(ControlModel):

    def __init__(self, num_assets, gamma=1.0, nu=0.01):
        super(ControlModel, CovarianceModel).__init__(self)
        self.num_assets = num_assets
        self.gamma = gamma
        self.nu = nu
        self.x = None
        self.x0 = None
        self.problem = None
        self._optima = None

    def run(self, data):
        x0, mu, sigma = data
        self.x0 = x0

        self.x = cvx.Variable(self.num_assets+1)

        objective = self.x.T*mu - self.gamma*cvx.quad_form(self.x, sigma)

        self.problem = cvx.Problem(cvx.Maximize(objective),
                           [
                               cvx.norm(self.x, 1) <= 1 - self.nu * cvx.norm(self.x - x0, 1),
                               self.x >= 0
                           ])
        self._optima = self.problem.solve()

    def optima(self):
        return self._optima

    def variables(self):
        return self.x.value.flatten()

    def get_input(self, L, past_data, ar_projections, ar_errors, bank_rate=1.0):
        proj = ar_projections.T[:,0]
        proj = np.concatenate((proj, np.array([bank_rate])))

        if len(ar_errors.shape) == 3:
            # janky way to tell what kind of error we have available.
            sigma_small = ar_errors[0]
            sigma = np.zeros(shape=(sigma_small.shape[0] + 1, sigma_small.shape[1] + 1))
            sigma[:-1, :-1] = sigma_small
            sigma[-1, -1] = 1.0
            return proj, sigma
        else:
            sigma = ar_errors.reshape(ar_errors.shape[0], 1)[:, 0]
            sigma = np.concatenate((sigma, np.array([0.0])))
            sigma = np.diag(sigma)
        return proj, sigma

    def apply_model_results(self, true_x):
        new_x = self.variables()
        updated_x = true_x.copy()
        # modify updated_x with buy/sell changes.
        sell, buy = np.maximum(0, -(new_x - true_x)), np.maximum(0, (new_x - true_x))
        updated_x = updated_x - sell + buy
        return updated_x


class MultiPeriodModel(ControlModel):
    """
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.559&rep=rep1&type=pdf
    page 4
    """
    def __init__(self, num_assets, L, theta, nu):
        self.L = L # planning horizon
        self.theta = theta # safety margin on std dev
        self.nu = nu # transaction cost
        self.num_assets = num_assets
        self.R = None
        self.xi = cvx.Variable((num_assets+1, L+1))
        self.eta = cvx.Variable((num_assets+1, L))
        self.zeta = cvx.Variable((num_assets+1, L))
        self.omega = cvx.Variable()
        self.problem = None
        self._optima = None

    def run(self, data):
        # x0 n x 1 initial state of portfolio,
        # returns n x L expected return at each time step,
        # sigmas n x n x L variance at each time step
        x0, returns, _ = data
        self.R = np.cumprod(returns, axis=1)
        # print("R:",self.R)
        objective = cvx.Maximize(self.omega)
        constraints = [self.omega <= self.R[:,self.L].T @ self.xi[:,-1],
                       self.zeta >= 0, self.xi >= 0, self.eta >= 0,
                       self.xi[:,0] == np.divide(x0, self.R[:,0]),
                       self.xi[-1,1:] == 0]
        A = (1 - self.nu) * self.R
        B = (1 + self.nu) * self.R
        for l in range(1, self.L + 1):
            # Equation 1.9
            constraints += [0 == -self.xi[:,l] + self.xi[:,l-1] - self.eta[:,l-1] + self.zeta[:,l-1],
                            0 <= A[:,l-1].T @ self.eta[:,l-1] - B[:,l-1] @ self.zeta[:,l-1]]
        self.problem = cvx.Problem(objective, constraints)
        self._optima = self.problem.solve()
        # print(self.problem.status)

    def optima(self):
        return self._optima

    def variables(self):
        zeta = self.zeta.value
        eta = self.eta.value
        xi = self.xi.value
        R = self.R
        return xi * R, eta * R[:,1:], zeta * R[:, 1:]

    def get_input(self, L, past_data, ar_projections, ar_errors, bank_rate=1.0):
        num_assets = self.num_assets

        ar_variances = np.zeros((num_assets, L))
        ar_variances[:, 1:] = np.repeat(ar_errors.reshape(-1, 1), L - 1, axis=1)
        variances = np.zeros((num_assets + 1, L))
        variances[:-1, :] = ar_variances[:, :]

        projections = np.ones((num_assets + 1, L)) * bank_rate
        projections[:-1, 0] = past_data.T[:, -1]
        projections[:-1, 1:] = ar_projections.T[:, :-1]
        # assert np.allclose(projections[-1, :], bank_rate)

        return projections, variances

    def apply_model_results(self, true_x):
        updated_x = true_x.copy()
        # print(true_x.shape)
        # modify updated_x with buy/sell changes.
        _, sell_all, buy_all = self.variables()
        sell, buy = sell_all[:, 0], buy_all[:, 0]
        updated_x = true_x - sell + buy
        return updated_x


class RobustMultiPeriodModel(ControlModel):
    """
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.559&rep=rep1&type=pdf
    page 15

    Note: Here we use theta as a scaling parameter on the variance, xi^T V xi,
    rather than on the standard deviation sqrt(xi^T V xi) to ease implementation in CVX
    """
    def __init__(self, num_assets, L, theta, nu):
        self.L = L # planning horizon
        self.theta = theta # safety margin on std dev
        self.nu = nu # transaction cost
        self.num_assets = num_assets
        self.R = None
        self.xi = cvx.Variable((num_assets+1, L+1))
        self.zeta = cvx.Variable((num_assets+1, L))
        self.eta = cvx.Variable((num_assets+1, L))
        self.omega = cvx.Variable()
        self.problem = None
        self._optima = None

    def run(self, data):
        # x0 n x 1 initial state of portfolio,
        # log_returns n x L expected log return at each time step,
        # sigmas n x L expected log variance at each time step
        x0, log_returns, log_vars = data

        # Expectations
        ExpR = np.exp(np.cumsum(log_returns + 0.5 * log_vars, axis=1))
        self.R = ExpR
        pl = [ np.concatenate([(1-self.nu) * ExpR[:,l], -(1+self.nu) * ExpR[:,l]]) for l in range(self.L) ]
        pLp1 = ExpR[:,-1]

        # Covariances
        VarR = (np.exp(np.cumsum(2 * log_returns + log_vars, axis=1)) * np.exp(np.cumsum(log_vars, axis=1)) -
                np.exp(np.cumsum(2 * log_returns + log_vars, axis=1)))
        Cl = [ np.diag(VarR[:,l]) for l in range(1, self.L+1) ]
        Vl = [ np.bmat( [[(1 - self.nu) ** 2 * C, -(1-self.nu) * (1 + self.nu) * C],
                         [-(1-self.nu) * (1 + self.nu) * C, (1 + self.nu) ** 2 * C]] ) for C in Cl ]
        VLp1 = Cl[-1]

        # print(Vl[0])
        # np.save("vl0", Vl[0])
        # print(sqrtm(Vl[0]))
        # print(np.linalg.norm(Vl[0]))

        objective = cvx.Maximize(self.omega)
        constraints = [self.omega <= pLp1 @ self.xi[:,self.L] - self.theta * cvx.norm(sqrtm(VLp1 + np.eye(VLp1.shape[0])*1e-8) @ self.xi[:,-1], 2),
                       self.xi >= 0, self.eta >= 0, self.zeta >= 0,
                       self.xi[:,0] == np.divide(x0, self.R[:,0]),
                       self.xi[-1,1:] == 0]
        alpha = (1 - self.nu) * self.R
        beta = (1 + self.nu) * self.R

        for l in range(1, self.L + 1):
            # Equation 1.9
            constraints += [0 == -self.xi[:,l] + self.xi[:,l-1] - self.eta[:,l-1] + self.zeta[:,l-1],
                            0 <= (alpha[:,l-1].T @ self.eta[:,l-1] - beta[:,l-1] @ self.zeta[:,l-1] -
                                  self.theta * cvx.norm(sqrtm(Vl[l-1] + np.eye(Vl[l-1].shape[0])*1e-8) @ cvx.bmat([[self.eta[:,l-1], self.zeta[:,l-1]]]).T, 2))
                            ]
        self.problem = cvx.Problem(objective, constraints)
        self._optima = self.problem.solve()
        print(self.problem.status)

    def optima(self):
        return self._optima

    def variables(self):
        zeta = self.zeta.value
        eta = self.eta.value
        xi = self.xi.value
        R = self.R
        return xi * R, eta * R[:,1:], zeta * R[:, 1:]

    def get_input(self, L, past_data, ar_projections, ar_errors, bank_rate=1.0):
        num_assets = self.num_assets

        ar_variances = np.zeros((num_assets, L-1))
        ar_variances[:, :] = np.repeat(ar_errors.reshape(-1, 1), L - 1, axis=1)

        # We want projections and variances in log space
        projections = np.zeros((num_assets + 1, L))
        projections[:-1, 0] = np.log(past_data.T[:, -1])
        # Technically, if you have normal data there would be a variance term here too,
        # but I'm not sure that makes sense for the autoregression
        projections[:-1, 1:] = np.log(ar_projections.T[:, :-1])

        variances = np.zeros((num_assets + 1, L))
        # This is a first-order approximation for normally distributed data, but it's the best we have for now
        variances[:-1, 1:] = ar_variances[:, :] / np.square(ar_projections.T[:, :-1])

        return projections, variances

    def apply_model_results(self, true_x):
        updated_x = true_x.copy()
        if self.problem.status != cvx.OPTIMAL:
            return updated_x
        # modify updated_x with buy/sell changes.
        _, sell_all, buy_all = self.variables()
        sell, buy = sell_all[:, 0], buy_all[:, 0]
        updated_x = updated_x - sell + buy
        return updated_x


def main():
    data = RealData()
    all_samples = data.sample()
    num_samples = 50
    num_assets = all_samples.shape[1]
    samples = all_samples[:num_samples]
    for i in range(samples.shape[1]):
        print(samples.T[i])

    L = 10
    ar = AutoRegression(L)
    ar.fit(samples)
    ar_projections, ar_errors = ar.predict(samples, L)
    projection_truth = all_samples[num_samples:num_samples+L]
    print("projection_errors", np.abs(projection_truth - ar_projections))

    # Run models
    mpc = MultiPeriodModel(num_assets, L-1, 2, .1)
    rmpc = RobustMultiPeriodModel(num_assets, L-1, 2, .1)

    projections, variances = mpc.get_input(L, samples, ar_projections, ar_errors)

    x0 = np.zeros((num_assets+1,))
    x0[-1] = 1.0

    mpc.run(data=(x0, projections, None))

    x, y, z = mpc.variables()
    print("x:",x)
    print("y:",y)
    print("z:",z)
    print(mpc.optima())

    rmpc.run(data=(x0, np.log(projections), variances))

    rx, ry, rz = rmpc.variables()
    print("rx:",rx)
    print("ry:",ry)
    print("rz:",rz)
    print(rmpc.optima())

    plt.plot(x.T,label='x')
    plt.plot(rx.T,':',label='rx')
    plt.legend()
    plt.show()

    plt.plot(y.T,label='y')
    plt.plot(ry.T,':',label='ry')
    plt.legend()
    plt.show()

    plt.plot(z.T,label='z')
    plt.plot(rz.T,':',label='rz')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    from data_models import GaussianNoise, NoisySine, RealData
    from prediction_models import UnbiasGaussianEstimator, AutoRegression

    # num_samples = 1000
    # num_assets = 3

    # mu_truth = np.ones(num_assets)
    # sigma_truth = np.diag([0.5, 0.3, 0.2])
    #
    # sampler = GaussianNoise()
    # data = np.zeros(shape=(num_samples, num_assets))
    #
    # for i in range(num_samples):
    #     data[i] = sampler.sample((mu_truth, sigma_truth))
    #
    # sample_mean, sample_covar = UnbiasGaussianEstimator().predict(data)
    #
    # for i in range(num_assets):
    #     print(sample_mean[i], sample_covar[i])

    # mpc = MultiPeriodModel(num_assets, 2, 2, .1)
    # x0 = np.ones((num_assets,)) / num_assets
    # sample_mean[0] = 1.1
    # sample_mean[1] = 0.9
    # means = np.repeat(sample_mean.reshape(-1, 1), 3, 1)
    # covs = np.repeat(sample_covar.reshape(-1, 1), 3, 1)
    # mpc.run(data=(x0, means, covs))
    #
    # x, y, z = mpc.variables()
    # print("x:", x)
    # print("y:", y)
    # print("z:", z)
    # print(mpc.optima())
    #
    # cov_model = CovarianceModel(num_assets=num_assets)
    # cov_model.run(data=(sample_mean, sample_covar), gamma=1.0)

    # print(cov_model.variables())
    # print(cov_model.optima())

    # noisy sine
    # data = NoisySine()
    # phase = np.array([1., .5, 2.])
    # noise = np.array([0.1, 0.03, 0.2])
    # samples = data.sample((phase, noise, 20))
    # for i in range(samples.shape[1]):
    #     print(samples.T[i])
    main()
