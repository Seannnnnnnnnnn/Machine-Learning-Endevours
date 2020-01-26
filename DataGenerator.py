"""
A python file for generating plots and data of a random variable
"""
import random
from typing import List
from matplotlib import pyplot as plt


def gen_data(n) -> List:
    m, c = random.random(), random.random()
    y = [m*x+c + random.gauss(0, 2) for x in range(n)]
    return y


def output(data):
    plt.plot(data, 'x')
    plt.title('Random Process')
    plt.show()


if __name__ == '__main__':
    process = gen_data(100)
    output(process)

