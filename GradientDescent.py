"""
an Implementation of the basic, gradient descent algorithm
"""
from Regression.Functions import cost_func, partial_theta1, partial_theta2
from Regression.DataGenerator import RandomProcess
from matplotlib import pyplot as plt
import numpy as np


def gradient_descent(theta1, theta2, data, alpha=0.0001):

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
    Y = RandomProcess()
    observations = Y.get_data()

    c, m = gradient_descent(0.5, 0.5, observations)
    print('c: ', c, 'm: ', m)
#    output = []
#    for _ in range(100):
#        output.append(m * _ + c)

    f = [observations[_][1] for _ in range(len(observations))]

    x = np.linspace(0,100,100)
    y = m*x+c
    plt.plot(f, 'ro')
    plt.plot(y)
    plt.title("sorry gradient descent machine broke")
    plt.show()

