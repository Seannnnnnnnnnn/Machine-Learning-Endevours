"""
an Implementation of the basic, gradient descent algorithm
"""
from Regression.Functions import cost_func, partial_theta1, partial_theta2
from Regression.DataGenerator import gen_data
from matplotlib import pyplot as plt
import numpy as np


def gradient_descent(theta1, theta2, data, alpha=0.000001):

    previous = 1
    correction = 1

    while correction > 1e-3:

        partial1 = partial_theta1(theta1, theta2, data)
        partial2 = partial_theta2(theta1, theta2, data)

        theta1 = theta1 - (alpha * partial1)
        theta2 = theta2 - (alpha * partial2)

        current = cost_func(theta1, theta2, data)
        correction = abs(previous - current)
        previous = current

        # debugging:
        print(theta1, theta2, current)

    return theta1, theta2


if __name__ == '__main__':
    observations = gen_data(100)

    c, m = gradient_descent(0.5, 0.5, observations)
    print('c: ', c, 'm: ', m)

    # generate a line to plot against our data
    x = np.linspace(0, 100, 100)
    y = m*x+c

    plt.plot(observations, 'ro')
    plt.plot(y)
    plt.title("Regression by Gradient Descent")
    plt.show()

