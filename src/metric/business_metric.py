# Standard imports
import numpy as np

# External imports
import matplotlib.pyplot as plt


#####################
# Private functions #
#####################

def _ecdf(data):
    """ Compute Empirical cumulative distribution function (ECDF)  """
    y = np.sort(data)
    n = y.size
    x = (np.arange(1, n+1) / n) * 100
    return(x,y)


#####################
# Exposed functions #
#####################


def business_metric(y_truth, y_pred, representative_value=None):
    """
    Calculate the business metric for each point.
    This is a regression metric.

    The metric correspond to the number of time there is the representative_value
    in the error. It is calculated like this: truth - pred / representative_value
    The representative_value is determined by the expert and make sense as a 
    measure of calculation of error.
    
    Parameters
    ----------
    y_truth: list(number)
        Truth values of the predicted variable
    y_pred: list(number)
        Predicted values of the predicted variable
        Each predicted value index should correspond to teh same point in the y_truth
    representative_value: number, optional
        Value that will become the unit of error, default is the standard deviation of y_truth
    
    Returns
    ----------
    list(float)
        Result of the metric for each point
    """
    if not representative_value:
        representative_value = np.std(y_truth)
    residual = np.abs([t - p for t, p in zip(y_truth, y_pred)])
    return list(residual / representative_value)


def get_important_quantiles(l):
    """
    Get the quantiles 5%, 25%, 50%, 77%, 95% for a list.

    Parameters
    ----------
    l: list(number)
        list of number

    Returns
    ----------
    dict
        contains the keys 0.05, 0.25, 0.50, 0.75, 0.95 with the
        corresponding quantile as value
    """
    return {
        "0.05": np.quantile(l, 0.05),
        "0.25": np.quantile(l, 0.25),
        "0.50": np.quantile(l, 0.5),
        "0.75": np.quantile(l, 0.75),
        "0.95": np.quantile(l, 0.95),
    }


def plot_business_metric(bm):
    """
    Plot the cumulative function of the business metric.
    With this plot we can say "in 90% of the cases we have an error of less
    the X representative_value".

    Parameters
    ----------
    bm: list(float)
        result of the function business_metric
    """
    bm = list(bm)
    x, y = _ecdf(bm)

    plt.plot(x, y)
    plt.vlines([5, 50, 95], ymin=min(bm), ymax=max(bm), color="red", linestyles="dashed")
    plt.ylabel("Number of representative value")
    plt.xlabel("% of value")
    plt.title("% of value that are below a certain number of representative value")
    plt.grid(True)

    plt.show()