import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt

import time

from data_models import *
from prediction_models import *
from control_models import *


def error(predicted_return, true_return):
    return (predicted_return - true_return)


def get_gaussian_data(num_samples, true_asset_value, asset_covariance, seed=1):
    num_assets = asset_covariance.shape[0]
    sampler = GaussianNoise(seed)
    data = np.zeros(shape=(num_samples, num_assets))
    for t in range(num_samples):
        sampler_input = (true_asset_value, asset_covariance)
        data[t] = sampler.sample(sampler_input)
    return data


def get_wiener_data(num_samples, true_asset_value, asset_covariance, seed=1):
    num_assets = asset_covariance.shape[0]
    steps = get_gaussian_data(num_samples, np.zeros((num_assets,)), asset_covariance, seed)
    return np.cumsum(steps, axis=0) + true_asset_value


def get_returns(data, investment_strategies, asset_predictions):
    num_samples = investment_strategies.shape[0]
    predicted_return = np.zeros(shape=(num_samples,))
    true_return = np.zeros(shape=(num_samples,))
    for t in range(num_samples):
        if t <= 2:
            continue
        observed_asset_value = data[t]
        predicted_asset_value = asset_predictions[t]
        investment_strategy = investment_strategies[t]
        true_return[t] = investment_strategy.dot(observed_asset_value)
        predicted_return[t] = investment_strategy.dot(predicted_asset_value)
    return predicted_return, true_return


def run_gaussian_norm(data, num_samples, num_assets, pred_params, control_params):
    gamma = control_params['gamma']
    regularization = control_params['regularization']

    prediction_model = UnbiasEstimator()
    window = pred_params['window']
    cov_model = NormModel(num_assets=num_assets, gamma=gamma, regularization=regularization)

    predicted_asset_values = np.zeros(shape=(num_samples, num_assets))
    investment_strategies = np.zeros(shape=(num_samples, num_assets))
    for t in range(num_samples):
        if t <= 2:
            continue
        if window is None:
            past_data = data[:t]
        else:
            past_data = data[max(0, t-window):t]
        predicted_asset_value, predicted_asset_variance = prediction_model.predict(past_data)
        predicted_asset_values[t] = predicted_asset_value

        control_input = (predicted_asset_value, predicted_asset_variance)
        cov_model.run(control_input)
        investment_strategy = cov_model.variables()
        investment_strategies[t] = investment_strategy

    return predicted_asset_values, investment_strategies


def run_gaussian_covar(data, num_samples, num_assets, pred_params, control_params):
    gamma = control_params['gamma']

    prediction_model = UnbiasEstimator()
    window = pred_params['window']
    cov_model = CovarianceModel(num_assets=num_assets, gamma=gamma)

    predicted_asset_values = np.zeros(shape=(num_samples, num_assets))
    investment_strategies = np.zeros(shape=(num_samples, num_assets))
    for t in range(num_samples):
        if t <= 2:
            continue
        if window is None:
            past_data = data[:t]
        else:
            past_data = data[max(0, t-window):t]
        predicted_asset_value, predicted_asset_variance = prediction_model.predict(past_data)
        predicted_asset_values[t] = predicted_asset_value

        control_input = (predicted_asset_value, predicted_asset_variance)
        cov_model.run(control_input)
        investment_strategy = cov_model.variables()
        investment_strategies[t] = investment_strategy

    return predicted_asset_values, investment_strategies


def run_gaussian_cvar(data, num_samples, num_assets, pred_params, control_params):
    beta = control_params['beta']
    epsilon = control_params['epsilon']

    prediction_model = UnbiasEstimator()
    window = pred_params['window']
    cvar_model = ValueAtRiskModel(num_assets=num_assets, beta=beta, epsilon=epsilon)

    predicted_asset_values = np.zeros(shape=(num_samples, num_assets))
    investment_strategies = np.zeros(shape=(num_samples, num_assets))
    for t in range(num_samples):
        if t <= 2:
            continue
        if window is None:
            past_data = data[:t]
        else:
            past_data = data[max(0, t-window):t]
        predicted_asset_value, predicted_asset_variance = prediction_model.predict(past_data)
        predicted_asset_values[t] = predicted_asset_value

        control_input = (predicted_asset_value, predicted_asset_variance)
        cvar_model.run(control_input)
        investment_strategy = cvar_model.variables()
        investment_strategies[t] = investment_strategy

    return predicted_asset_values, investment_strategies

def run_simple_gaussian_experiments(params, plot=False, seed=1):
    num_samples = 100
    true_asset_value = params['asset_value']
    asset_covariance = params['asset_covariance']
    gamma = params['gamma']
    beta = params['beta']
    epsilon = params['epsilon']
    window = params['window']
    data = get_gaussian_data(num_samples, true_asset_value, asset_covariance, seed)
    data = np.clip(data, 1e-3, None)
    num_assets = data.shape[1]

    if plot:
        plt.plot(data, label='Asset Values')
        plt.legend()
        plt.title('Input Data')
        plt.show()

    # Add experiments to run here.
    experiments = [
        ("gaussian_unbiased_covar", run_gaussian_covar, {'window': None}, {"gamma": gamma}),
        ("gaussian_unbiased_l1", run_gaussian_norm, {'window': None}, {"gamma": gamma, "regularization": 1}),
        ("gaussian_unbiased_l2", run_gaussian_norm, {'window': None}, {"gamma": gamma, "regularization": 2}),
        ("gaussian_unbiased_cvar", run_gaussian_cvar, {'window': None}, {"beta": beta, "epsilon": epsilon}),
        ("gaussian_windowed_covar", run_gaussian_covar, {'window': window}, {"gamma": gamma}),
        ("gaussian_windowed_l1", run_gaussian_norm, {'window': window}, {"gamma": gamma, "regularization": 1}),
        ("gaussian_windowed_l2", run_gaussian_norm, {'window': window}, {"gamma": gamma, "regularization": 2}),
        ("gaussian_windowed_cvar", run_gaussian_cvar, {'window': window}, {"beta": beta, "epsilon": epsilon}),
    ]

    bar_plot_mean = []
    bar_plot_std = []

    results = {}
    results['true_values'] = data
    for name, experiment_func, pred_params, control_params in experiments:
        predicted_asset_values, investment_strategies = experiment_func(data,
                                                                        num_samples,
                                                                        num_assets,
                                                                        pred_params,
                                                                        control_params)
        predicted_return, true_return = get_returns(data, investment_strategies, predicted_asset_values)
        results[name] = {}
        results[name]['predicted_return'] = predicted_return
        results[name]['strategies'] = investment_strategies
        results[name]['predicted_values'] = predicted_asset_values
        results[name]['true_return'] = true_return
        print(name, np.sum(true_return))
        bar_plot_mean.append(np.mean(true_return))
        bar_plot_std.append(np.std(true_return))
        # all_error = error(predicted_return, true_return)
        # window = 10
        # for i in range(0, num_samples-window, window):
        #     print(name, np.mean(all_error[i:i + window]))
        if plot:
            # We really just care about how well the investment strategies actually do,
            # which is given by true_return.
            plt.plot(np.arange(3, num_samples), true_return[3:], label=name + ' true return', alpha=0.33)
            # In final plots, predicted return may not be relevant.
            plt.plot(np.arange(3, num_samples), predicted_return[3:], label=name + ' predicted return')
    if plot:
        plt.legend()
        plt.show()

        plt.bar(np.arange(len(experiments)), height=bar_plot_mean, yerr=bar_plot_std)
        plt.show()

    return results


def run_ltv_gaussian_experiments(params, plot=False, seed=1):
    num_samples = 100
    true_asset_v0 = params['asset_value']
    true_asset_delta = params['asset_delta']
    asset_covariance = params['asset_covariance']
    gamma = params['gamma']
    beta = params['beta']
    epsilon = params['epsilon']
    window = params['window']
    num_assets = true_asset_v0.shape[0]
    true_asset_value = true_asset_v0 + (true_asset_delta.T @ np.arange(0,num_samples).reshape(-1,1).T).T
    data = get_gaussian_data(num_samples, np.zeros((num_assets,)), asset_covariance, seed) + true_asset_value
    data = np.clip(data, 1e-3, None)

    if plot:
        plt.plot(data, label='Asset Values')
        plt.legend()
        plt.title('Input Data')
        plt.show()

    # Add experiments to run here.
    experiments = [
        ("gaussian_unbiased_covar", run_gaussian_covar, {'window': None}, {"gamma": gamma}),
        ("gaussian_unbiased_l1", run_gaussian_norm, {'window': None}, {"gamma": gamma, "regularization": 1}),
        ("gaussian_unbiased_l2", run_gaussian_norm, {'window': None}, {"gamma": gamma, "regularization": 2}),
        ("gaussian_unbiased_cvar", run_gaussian_cvar, {'window': None}, {"beta": beta, "epsilon": epsilon}),
        ("gaussian_windowed_covar", run_gaussian_covar, {'window': window}, {"gamma": gamma}),
        ("gaussian_windowed_l1", run_gaussian_norm, {'window': window}, {"gamma": gamma, "regularization": 1}),
        ("gaussian_windowed_l2", run_gaussian_norm, {'window': window}, {"gamma": gamma, "regularization": 2}),
        ("gaussian_windowed_cvar", run_gaussian_cvar, {'window': window}, {"beta": beta, "epsilon": epsilon}),
    ]

    bar_plot_mean = []
    bar_plot_std = []

    results = {}
    results['true_values'] = data
    for name, experiment_func, pred_params, control_params in experiments:
        predicted_asset_values, investment_strategies = experiment_func(data,
                                                                        num_samples,
                                                                        num_assets,
                                                                        pred_params,
                                                                        control_params)
        predicted_return, true_return = get_returns(data, investment_strategies, predicted_asset_values)
        results[name] = {}
        results[name]['predicted_return'] = predicted_return
        results[name]['strategies'] = investment_strategies
        results[name]['predicted_values'] = predicted_asset_values
        results[name]['true_return'] = true_return
        print(name, np.sum(true_return))
        bar_plot_mean.append(np.mean(true_return))
        bar_plot_std.append(np.std(true_return))
        # all_error = error(predicted_return, true_return)
        # window = 10
        # for i in range(0, num_samples-window, window):
        #     print(name, np.mean(all_error[i:i + window]))
        if plot:
            # We really just care about how well the investment strategies actually do,
            # which is given by true_return.
            plt.plot(np.arange(3, num_samples), true_return[3:], label=name + ' true return', alpha=0.33)
            # In final plots, predicted return may not be relevant.
            plt.plot(np.arange(3, num_samples), predicted_return[3:], label=name + ' predicted return')
    if plot:
        plt.legend()
        plt.show()

        plt.bar(np.arange(len(experiments)), height=bar_plot_mean, yerr=bar_plot_std)
        plt.show()

    return results


def run_wiener_experiments(params, plot=False, seed=1):
    num_samples = 100
    true_asset_v0 = params['asset_value']
    asset_covariance = params['asset_covariance']
    gamma = params['gamma']
    beta = params['beta']
    epsilon = params['epsilon']
    window = params['window']
    data = get_wiener_data(num_samples, true_asset_v0, asset_covariance, seed)
    data = np.clip(data, 1e-3, None)
    num_assets = data.shape[1]

    if plot:
        plt.plot(data, label='Asset Values')
        plt.legend()
        plt.title('Input Data')
        plt.show()

    # Add experiments to run here.
    experiments = [
        ("gaussian_unbiased_covar", run_gaussian_covar, {'window': None}, {"gamma": gamma}),
        ("gaussian_unbiased_l1", run_gaussian_norm, {'window': None}, {"gamma": gamma, "regularization": 1}),
        ("gaussian_unbiased_l2", run_gaussian_norm, {'window': None}, {"gamma": gamma, "regularization": 2}),
        ("gaussian_unbiased_cvar", run_gaussian_cvar, {'window': None}, {"beta": beta, "epsilon": epsilon}),
        ("gaussian_windowed_covar", run_gaussian_covar, {'window': window}, {"gamma": gamma}),
        ("gaussian_windowed_l1", run_gaussian_norm, {'window': window}, {"gamma": gamma, "regularization": 1}),
        ("gaussian_windowed_l2", run_gaussian_norm, {'window': window}, {"gamma": gamma, "regularization": 2}),
        ("gaussian_windowed_cvar", run_gaussian_cvar, {'window': window}, {"beta": beta, "epsilon": epsilon}),
    ]

    bar_plot_mean = []
    bar_plot_std = []
    results = {}
    results['true_values'] = data
    for name, experiment_func, pred_params, control_params in experiments:
        predicted_asset_values, investment_strategies = experiment_func(data,
                                                                        num_samples,
                                                                        num_assets,
                                                                        pred_params,
                                                                        control_params)
        predicted_return, true_return = get_returns(data, investment_strategies, predicted_asset_values)
        results[name] = {}
        results[name]['predicted_return'] = predicted_return
        results[name]['strategies'] = investment_strategies
        results[name]['predicted_values'] = predicted_asset_values
        results[name]['true_return'] = true_return
        print(name, np.sum(true_return))
        bar_plot_mean.append(np.mean(true_return))
        bar_plot_std.append(np.std(true_return))
        # all_error = error(predicted_return, true_return)
        # window = 10
        # for i in range(0, num_samples-window, window):
        #     print(name, np.mean(all_error[i:i + window]))
        if plot:
            # We really just care about how well the investment strategies actually do,
            # which is given by true_return.
            plt.plot(np.arange(3, num_samples), true_return[3:], label=name + ' true return', alpha=0.33)
            # In final plots, predicted return may not be relevant.
            plt.plot(np.arange(3, num_samples), predicted_return[3:], label=name + ' predicted return')
    if plot:
        plt.legend()
        plt.show()

        plt.bar(np.arange(len(experiments)), height=bar_plot_mean, yerr=bar_plot_std)
        plt.show()

    return results


if __name__ == "__main__":
    run_simple_gaussian_experiments(params={'asset_value': np.array([0.8, 1.0, 1.1]),
                                           'asset_covariance': np.diag([0.02, 0.01, 0.03]),
                                           'gamma': 1,
                                           'beta': 0.0,
                                           'epsilon': 0.05,
                                           'window': 10},
                                    plot=True, seed=int(time.time()))
    run_ltv_gaussian_experiments(params={'asset_value': np.array([0.9, 1.2, 1.0]),
                                        'asset_covariance': np.diag([1.0, 1.0, 0.2]) * 0.02,
                                        'asset_delta': np.array([[0.002, -0.003, 0.001]]),
                                        'gamma': 1,
                                        'beta': 0.0,
                                        'epsilon': 0.05,
                                        'window': 10},
                                 plot=True, seed=int(time.time()))
    run_wiener_experiments(params={'asset_value': np.array([0.9, 1.2, 1.0]),
                                  'asset_covariance': np.diag([1.0, 1.0, 0.2]) * 0.02,
                                  'gamma': 1,
                                  'beta': 0.0,
                                  'epsilon': 0.05,
                                  'window': 10},
                           plot=True, seed=int(time.time()))
