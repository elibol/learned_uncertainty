import pickle

import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt


def plot_residuals(runs, model_name):
    # Plot residual of predicted vs actual return
    for r in runs:
        plt.plot(r['gaussian_unbiased_covar']['predicted_return'] - r['gaussian_unbiased_covar']['true_return'], label='gaussian_unbiased_covar', alpha=0.25)
        plt.plot(r['gaussian_unbiased_l1']['predicted_return'] - r['gaussian_unbiased_l1']['true_return'], label='gaussian_unbiased_l1', alpha=0.25)
        plt.plot(r['gaussian_unbiased_l2']['predicted_return'] - r['gaussian_unbiased_l2']['true_return'], label='gaussian_unbiased_l2', alpha=0.25)
        plt.plot(r['gaussian_windowed_covar']['predicted_return'] - r['gaussian_windowed_covar']['true_return'], label='gaussian_windowed_covar', alpha=0.25)
        plt.plot(r['gaussian_windowed_l1']['predicted_return'] - r['gaussian_windowed_l1']['true_return'], label='gaussian_windowed_l1', alpha=0.25)
        plt.plot(r['gaussian_windowed_l2']['predicted_return'] - r['gaussian_windowed_l2']['true_return'], label='gaussian_windowed_l2', alpha=0.25)
    plt.ylabel("Predicted - Actual Return")
    plt.xlabel("Time Step")
    plt.title("Return Residual for {} Model".format(model_name))
    plt.savefig("out/return_residual-{}_model.png".format(model_name))
    plt.show()


def plot_runs(runs, model_name):
    # plot_residuals(runs, model_name)
    plot_gamma_sweep(runs, model_name)


def main():
    with open("simple_gauss.pkl", 'rb') as f:
        simple_gauss = pickle.load(f)
    plot_runs(simple_gauss, 'Gaussian')
    with open("ltv_gauss.pkl", 'rb') as f:
        ltv_gauss = pickle.load(f)
    plot_runs(ltv_gauss, 'Trending-Gaussian')
    with open("wiener.pkl", 'rb') as f:
        wiener = pickle.load(f)
    plot_runs(wiener, 'Wiener-Process')


if __name__ == "__main__":
    main()