# import autograd as ag
# import autograd.numpy as np
import numpy as np
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
plt.rc('figure', figsize=[10, 6])


class PredictionModel(object):
    """
    Models used to predict asset values based on past observations.
    """

    def __init__(self):
        pass

    def fit(self, samples):
        pass

    def predict(self, samples, n):
        pass


class UnbiasGaussianEstimator(PredictionModel):

    def __init__(self):
        super(UnbiasGaussianEstimator, self).__init__()

    def sample_mean(self, samples):
        return np.mean(samples, axis=0)

    def sample_covariance(self, samples):
        return np.cov(samples, rowvar=False)

    def predict(self, samples, n=1, assume_diag=True):
        projected_means = np.empty(shape=(n, samples.shape[1]))
        projected_covariance = np.empty(shape=(n, samples.shape[1], samples.shape[1]))
        projected_samples = samples
        for i in range(n):
            projected_means[i] = self.sample_mean(projected_samples)
            covar = self.sample_covariance(projected_samples)
            if covar.shape == ():
                covar = covar.reshape(1, 1)
            if assume_diag:
                covar = np.diag(np.diag(covar))
            projected_covariance[i, :, :] = covar
            projected_samples = np.vstack((projected_samples, projected_means[i]))

        if n == 1:
            return projected_means[0], projected_covariance[0]
        else:
            return projected_means, projected_covariance


class LogNormalMLE(UnbiasGaussianEstimator):

    def __init__(self):
        # https://www.wikiwand.com/en/Log-normal_distribution#/Maximum_likelihood_estimation_of_parameters
        super(LogNormalMLE, self).__init__()

    def sample_mean(self, samples):
        return np.mean(np.log(samples), axis=0)

    def sample_covariance(self, samples):
        return np.sqrt(np.cov(np.log(samples), rowvar=False, bias=True))


class AutoRegressionBase(PredictionModel):

    def __init__(self, p, predictor_cls, regularizer=0.001):
        super(AutoRegressionBase, self).__init__()
        self.p = p
        self.data = None
        self.models = None
        self.regularizer = regularizer
        self.predictor_cls = predictor_cls

    def fit(self, samples):
        self.data = samples.T
        self.models = []
        for i in range(self.data.shape[0]):
            model = self.predictor_cls(self.p, self.regularizer)
            model.fit(self.data[i])
            self.models.append(model)

    def predict(self, samples, n):
        predictions = np.empty(shape=(n, samples.shape[1]))
        errors = np.empty(shape=(samples.shape[1],))
        for i in range(samples.shape[1]):
            predictions[:, i], error = self.models[i].predict(samples.T[i], n)
            errors[i] = error
        return predictions, errors


class AutoRegression(AutoRegressionBase):

    def __init__(self, p, regularizer=0.001):
        super(AutoRegression, self).__init__(p,
                                             predictor_cls=OneDimensionalAutoRegression,
                                             regularizer=regularizer)


class LogAutoRegression(AutoRegressionBase):

    def __init__(self, p, regularizer=0.001):
        super(LogAutoRegression, self).__init__(p,
                                                predictor_cls=OneDimensionalLogAutoRegression,
                                                regularizer=regularizer)


class OneDimensionalAutoRegression(object):

    def __init__(self, p, regularizer=0.001):
        self.w = None
        self.p = p
        self.regularizer = regularizer

    def to_mat(self, samples):
        X = np.zeros([samples.shape[0]-self.p, self.p])
        for i in range(samples.shape[0] - self.p):
            X[i] = samples[i:i+self.p]
        y = samples[self.p:]
        return X, y

    def fit(self, samples):
        assert(samples.shape[0] > self.p), "Number of samples are less than autoregression length"
        X, y = self.to_mat(samples)
        self.regularizer = 0.0001
        self.w = np.matmul(\
                np.linalg.inv(np.matmul(X.transpose(),X) + self.regularizer * np.identity(X.shape[1])),\
                np.matmul(X.transpose(),y))

    def predict(self, r, n):
        assert(r.shape[0] >= self.p), "Need at least " + str(p) + " points to estimate uncertainty"
        uX, uY = self.to_mat(r)
        err = np.max(np.abs(np.matmul(uX, self.w) - uY))
        plen = self.w.shape[0]
        pred = np.zeros([n + plen])
        pred[0:plen] = r[-plen:]
        for i in range(n):
            pred[plen + i] = np.dot(pred[i:i+plen], self.w)
        return pred[-n:], err


class OneDimensionalLogAutoRegression(OneDimensionalAutoRegression):

    def __init__(self, p, regularizer=0.001):
        super(OneDimensionalLogAutoRegression, self).__init__(p,
                                                              regularizer=regularizer)
    def fit(self, samples):
        assert(samples.shape[0] > self.p), "Number of samples are less than autoregression length"
        X, y = self.to_mat(samples)
        X = np.log(X)
        y = np.log(y)
        self.regularizer = 0.0001
        self.w = np.matmul(\
                np.linalg.inv(np.matmul(X.transpose(),X) + self.regularizer * np.identity(X.shape[1])),\
                np.matmul(X.transpose(),y))

    def predict(self, r, n):
        assert(r.shape[0] > self.p + 10), "Need at least 10 points to estimate uncertainty"
        uX, uY = self.to_mat(r)
        uX = np.log(uX)
        uY = np.log(uY)
        err = np.max(np.abs(np.matmul(uX, self.w) - uY))
        plen = self.w.shape[0]
        pred = np.zeros([n + plen])
        pred[0:plen] = r[-plen:]
        for i in range(n):
            pred[plen + i] = np.dot(pred[i:i+plen], self.w)
        return pred[-n:], err



def test_autoregress(noise = 0.1):
    num_samples = 1000
    sine_samples = 0.1 * np.sin(np.linspace(0, 10 * np.pi, num_samples)) + 2
    reg_len = 40
    sine_samples += np.random.normal(0, noise, sine_samples.shape[0])
    ar = OneDimensionalAutoRegression(reg_len)
    ar.fit(sine_samples)

    num_project = 500
    pred_sine, err = ar.predict(sine_samples, num_project)
    plt.plot(np.arange(num_samples), sine_samples)
    plt.plot(np.arange(num_samples-1, num_samples+num_project-1), pred_sine, linewidth = 1.0 + 5.0*err, alpha = 0.5)
    plt.savefig('test_plots/test_regression.png')
    plt.close()
    print(err)


if __name__ == "__main__":
    from data_models import GaussianNoise

    # import pdb; pdb.set_trace()
    # test_autoregress()

    num_samples = 1000
    num_assets = 3

    mu_truth = np.ones(num_assets)
    sigma_truth = np.diag([0.1, 0.03, 0.2])
    sampler = GaussianNoise()

    data = np.zeros(shape=(num_samples, num_assets))
    for i in range(num_samples):
        data[i] = sampler.sample((mu_truth, sigma_truth))
    data = np.clip(data, 0.0001, None)

    L = 3
    ar = AutoRegression(L)
    ar.fit(data)
    predictions, errors = ar.predict(data, L)
    for l in range(L):
        print("\n prediction", l, predictions[l], errors[l])

    L = 3
    arlog = LogAutoRegression(L)
    arlog.fit(data)
    predictions, errors = arlog.predict(data, L)
    for l in range(L):
        print("\n prediction", l, predictions[l], errors[l])

    # num_samples = 1000
    # num_assets = 3
    #
    # mu_truth = np.ones(num_assets)
    # sigma_truth = np.diag([0.5, 0.3, 0.2])
    # sampler = GaussianNoise()
    #
    # data = np.zeros(shape=(num_samples, num_assets))
    # for i in range(num_samples):
    #     data[i] = sampler.sample((mu_truth, sigma_truth))
    #
    # L = 3
    # sample_mean, sample_variance = UnbiasGaussianEstimator().predict(data, L)
    # for l in range(L):
    #     print("\n projection", l)
    #     for i in range(num_assets):
    #         print(sample_mean[l][i], sample_variance[l][i])

    # log normal test.
    num_samples = 100000
    samples = np.random.lognormal(123, 67, num_samples).reshape(num_samples, 1)
    ln_mle = LogNormalMLE()
    mean, var = ln_mle.predict(samples)
    print(mean, var)
