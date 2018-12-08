import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
plt.rc('figure', figsize=[10, 6])

import time

from data_models import *
from prediction_models import *
from control_models import *

USE_RAY = False
if USE_RAY:
    import ray


def get_real_data(max_assets):
    sampler = RealData()
    return sampler.labels()[:max_assets], sampler.dates()[:max_assets], sampler.sample()[:, :max_assets]


def run_real_mpc(name, data, num_samples, num_assets, pred_cls, pred_params, ctrl_cls, control_params, bank_rate, L):
    print("running", name, pred_cls, ctrl_cls)
    augmented_data = np.hstack((data, np.ones((data.shape[0], 1)) * bank_rate))

    prediction_model = pred_cls(**pred_params)

    ctrl_model = ctrl_cls(**control_params)

    investment_strategies = np.zeros(shape=(num_samples, num_assets + 1))
    net_values = np.zeros(shape=(num_samples, num_assets + 1))
    current_x = np.zeros(shape=(num_assets + 1,))
    current_x[-1] = 1.0
    for t in range(num_samples):
        if t <= L + 20:
            net_values[t] = current_x
            continue
        past_data = data[:t]
        prediction_model.fit(past_data)
        predicted_return, predicted_return_error = prediction_model.predict(past_data, L)
        ctrl_proj, ctrl_var = ctrl_model.get_input(L, past_data, predicted_return, predicted_return_error, bank_rate)

        control_inputs = (current_x, ctrl_proj, ctrl_var)
        ctrl_model.run(control_inputs)

        observed_returns = augmented_data[t]
        investment_strategy = ctrl_model.apply_model_results(current_x)
        investment_strategies[t] = investment_strategy

        current_x = observed_returns * investment_strategy
        net_values[t] = current_x
        if t % 10 == 0:
            print(name, "t", t, "sum(current_x)", np.sum(current_x))

    return net_values, investment_strategies


def run_real_experiments(L, bank_rate, plot=False, seed=1):
    max_assets = 30
    data_labels, data_dates, data = get_real_data(max_assets)
    print("date range:", data_dates[0][0], "-", data_dates[0][-1])
    num_samples = data.shape[0]
    num_assets = data.shape[1]

    if plot:
        for i in range(num_assets):
            plt.plot(data.T[i], label=data_labels[i])
        plt.legend()
        plt.title('Input Data')
        plt.savefig("figures/real_data/input_data.png")

    # Add experiments to run here.
    experiments = [
        # multiperiod with autoregressive
        ("real_ar_mpm", run_real_mpc,
         AutoRegression, {"p": L, "regularizer": 0.001},
         MultiPeriodModel, {"num_assets": num_assets, "L": L - 1, "theta": 2.0, "nu": 0.01},),

        # multiperiod robust with autoregressive
        # ("real_ar_mpm_robust", run_real_mpc,
        #  AutoRegression, {"p": L, "regularizer": 0.001},
        #  RobustMultiPeriodModel, {"num_assets": num_assets, "L": L - 1, "theta": 0.0, "nu": 0.01},),

        # # L1-norm model with unbias gaussian
        # ("real_unbias_norm", run_real_mpc,
        #  UnbiasGaussianEstimator, {},
        #  NormModel, {"num_assets": num_assets, "gamma": 1.0, "norm": 2, "nu": 0.01},),

        # L1-norm model with unbias gaussian
        # ("real_ar_norm", run_real_mpc,
        #  AutoRegression, {"p": L, "regularizer": 0.1},
        #  NormModel, {"num_assets": num_assets, "gamma": 1.0, "norm": 2, "nu": 0.01},),

        # Weighted-norm model with unbias gaussian
        ("real_unbias_weighted", run_real_mpc,
         UnbiasGaussianEstimator, {},
         CovarianceModel, {"num_assets": num_assets, "gamma": 1.0, "nu": 0.01},),

        # Weighted-norm model with autoregression
        ("real_ar_weighted", run_real_mpc,
         AutoRegression, {"p": L, "regularizer": 0.001},
         CovarianceModel, {"num_assets": num_assets, "gamma": 1.0, "nu": 0.01},),
    ]

    results = {}
    for name, experiment_func, pred_cls, pred_params, ctrl_cls, control_params in experiments:
        money_values, investment_strategies = experiment_func(name,
                                                              data,
                                                              num_samples,
                                                              num_assets,
                                                              pred_cls,
                                                              pred_params,
                                                              ctrl_cls,
                                                              control_params,
                                                              bank_rate, L)
        results[name] = {}
        results[name]['money_values'] = money_values
        results[name]['strategies'] = investment_strategies
        print(name, "total money value", np.sum(money_values[-1]))
        np.save(name + "-portfolio_values", money_values)
        np.save(name + '-stategies', investment_strategies)

        plt.figure()
        for i in range(num_assets):
            plt.plot(investment_strategies[:, i], label=data_labels[i], alpha=0.5)
        plt.legend()
        plt.title("Investment Strategy ("+name+")")
        plt.savefig("figures/real_data/investment_strat_"+name+".png")

    if plot:
        plt.figure()
        for name in results:
            money_values = results[name]['money_values']
            total_values = np.sum(money_values, axis=1)
            plt.plot(total_values, label=name, alpha=0.5)
        plt.legend()
        plt.title("Money Comparison")
        plt.savefig("figures/real_data/money_comparison.png")

    return results


if __name__ == "__main__":
    if USE_RAY:
        ray.init()
    run_real_experiments(L=4,
                         bank_rate=1.0,
                         plot=True, seed=int(time.time()))
