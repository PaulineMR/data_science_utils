# Standard imports
import numpy as np

# External imports
import matplotlib.pyplot as plt

#####################
# Private functions #
#####################
def _ecdf(data):
    """
    Compute Empirical cumulative distribution function (ECDF)
    """
    y = np.sort(data) n = y.size x = (np.arange(1, n+1) / n) * 100
    return(x,y)

#####################
# Exposed functions #
#####################

def business_metric(y_truth, y_pred, representative_value=None):
    """
    """
    if not representative_value:
        representative_value = np.std(y_truth)
    residual = np.abs([t - p for t, p in zip(y_truth, y_pred)])
    return residual / representative_value

def get_quantile_business_metric(bm): 
    """
    """ 
    return { "0.05": np.quantile(bm, 0.05), "0.25": np.quantile(bm, 0.25), "0.50": np.quantile(bm, 0.5), "0.75": np.quantile(bm, 0.75), "0.95": np.quantile(bm, 0.95), }

def plot_business_metric(bm):
    """
    Cumulative dans 90% des cas on a une erreur de au moins 3 valeurs représenttive
    """
    bm = list(bm) x, y = _ecdf(bm)
    plt.plot(x, y) plt.vlines([5, 50, 95], ymin=min(bm), ymax=max(bm), color="red", linestyles="dashed") plt.ylabel("Number of representative value") plt.xlabel("% of value") plt.title("% of value that are below a certain number of representative value") plt.grid(True)
    plt.show()