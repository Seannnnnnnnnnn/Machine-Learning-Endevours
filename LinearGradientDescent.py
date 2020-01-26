"""
an Implementation of the basic, gradient descent algorithm
"""
from LinearRegression.Functions import cost_func, partial_theta0, partial_theta1
from LinearRegression.DataGenerator import gen_data_frame
from matplotlib import pyplot as plt
import random


def gradient_descent(data, alpha=1e-3, threshold=1e-2):

    previous = 1
    correction = 1
    theta0 = random.uniform(0, 1)
    theta1 = random.uniform(0, 1)

    while correction > threshold:

        partial1 = partial_theta0(theta0, theta1, data)
        partial2 = partial_theta1(theta0, theta1, data)

        theta0 = theta0 - (alpha * partial1)
        theta1 = theta1 - (alpha * partial2)

        current = cost_func(theta0, theta1, data)
        correction = abs(previous - current)
        previous = current

        # debugging:
        print(theta0, theta1, current)

    return theta0, theta1


if __name__ == '__main__':
    observations = gen_data_frame(100)

    theta_0, theta_1 = gradient_descent(observations)
    print('c: ', theta_0, 'm: ', theta_1)

    # generate a line to plot against our data
    y_hat = [theta_1 * observations['x'][i] + theta_0 for i in range(len(observations.index))]

    plt.scatter(observations['x'], observations['y'])
    plt.plot(observations['x'], y_hat, '-r')
    plt.title("Linear Regression by Gradient Descent")
    plt.show()
