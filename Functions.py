"""
implementation of the cost function and its partial derivatives
"""
import numpy as np


def partial_theta0(theta0, theta1, data):
    sum_component, n = 0, len(data.index)
    y = data['y']
    x = data['x']

    for i in range(len(data.index)):
        sum_component += theta0 + theta1 * x[i] - y[i]
    return sum_component / n


def partial_theta1(theta0, theta1, data):
    sum_component, n = 0, len(data.index)
    y = data['y']
    x = data['x']

    for i in range(len(data.index)):
        sum_component += (theta0 + theta1 * x[i] - y[i]) * x[i]
    return sum_component / n


def cost_func(theta0, theta1, data: pd.DataFrame):
    sum_component, n = 0, len(data.index)
    y = data['y']
    x = data['x']

    for i in range(len(data.index)):
        sum_component += (theta0 + theta1 * x[i] - y[i])**2

    return sum_component / 2*n


def logistic_cost_func(theta0, theta1, data):
    sum_component, n = 0, len(data.index)
    y = data['y']
    x = data['x']

    for i in range(len(data.index)):
        sum_component += y[i]*np.log(sigmoid(theta0 + theta1*x[i])) + (1-y[i])*np.log(1-sigmoid(theta0 + theta1*x[i]))

    return sum_component / n


def hypothesis(theta0, theta1, data):
    x = data['x']
    y = data['y']
    h = [1 / (1 + np.exp(-(theta0 + theta1 * x[i]))) for i in range(len(data.index))]
    return h

