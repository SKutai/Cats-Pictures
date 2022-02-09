import numpy as np
from layers import *

def logistic_est_loss(y_est, y):
    """
    Compute the logistic loss

    Parameters
    ----------
    y_est: float
        Input to logistic function
    y: float
        Target value
    
    Returns
    -------
    float: Logistic loss over all samples
    """
    return -y*np.log(y_est) - (1-y)*np.log(1-y_est)


def logistic_loss_deriv(u, y):
    """
    Compute the gradient of the logistic loss over many samples
    with respect to its inputs

    Parameters
    ----------
    u: ndarray(N)
        Input to logistic function
    y: float
        Target value
    
    Returns
    -------
    float:
        Derivative of logistic loss over all samples
    """
    return logistic(u)-y


def logistic_est_loss_deriv(y_est, y):
    """
    Compute the derivative of the logistic loss with
    respect to its input variable, given only the 
    output y_est of the logistic function

    Parameters
    ----------
    y_est: float
        Estimated output of the logistic function
    y: float
        Target value
    
    Returns
    -------
    ndarray(N):
        Derivative of logistic loss over all samples
    """
    return y_est-y

def softmax_est_crossentropy_loss(y_est, y):
    """
    Compute the multiclass softmax cross-entropy loss
    given the output of the softmax function

    Parameters
    ----------
    y_est: ndarray(N)
        Output of the softmax function
    y: ndarray(N)
        Target values
    
    Returns
    -------
    float: Cross-entropy
    """
    return -np.sum(y*np.log(y_est))

def softmax_est_crossentropy_deriv(y_est, y):
    """
    Compute the gradient of the multiclass softmax cross-entropy
    with respect to its input variables, given only the output
    of the softmax function

    Parameters
    ----------
    y_est: ndarray(N)
        Output of the softmax function
    y: ndarray(N)
        Target values
    
    Returns
    -------
    ndarray(N):
        Derivative of multiclass softmax cross-entropy
    """
    return y_est - y