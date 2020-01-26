from typing import List


def partial_theta1(theta1, theta2, data: List[tuple]):
    sum_component, n = 0, len(data)

    for tup in data:

        x = tup[0]
        y = tup[1]

        sum_component += theta1 + theta2 * x - y

    return sum_component / n


def partial_theta2(theta1, theta2, data: List[tuple]):
    sum_component, n = 0, len(data)

    for tup in data:
        x = tup[0]
        y = tup[1]

        sum_component += (theta1 + theta2*x - y)*x

    return sum_component / n


def cost_func(theta1, theta2, data: List[tuple]):
    sum_component, n = 0, len(data)

    for tup in data:
        x = tup[0]
        y = tup[1]

        sum_component += (theta1 + theta2*x - y)**2

    return sum_component / 2*n
