"""
A python file for generating plots and data of a random variable
TODO; REWRITE FUNCTIONALLY & SUBMIT TO GITHUB REPO
"""
import random
from typing import List
from matplotlib import pyplot as plt


class RandomProcess:

    def __init__(self):
        self.data = []
        self.time = None

    def __simulate(self, n):
        """
        simulates N observations of the process
        """
        gradient = random.random()
        intercept = random.random()

        print('Gradient: ', gradient)
        print('Intercept: ', intercept)

        for x in range(n):

            error = random.gauss(0, 5)
            self.data.append((x, (gradient*x + intercept + error)))

        self.time = [_ for _ in range(n)]

    def get_data(self):
        self.__simulate(100)
        return self.data

    def output(self):
        if not self.time:
            self.__simulate(100)
        plt.plot(self.data, 'x')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


def gen_data(n) -> List:
    m, c = random.random(), random.random()
    y = [m*x+c + random.gauss(0, 5) for x in range(n)]
    return y


def output(data):
    plt.plot(data, 'x')
    plt.title('Random Process')
    plt.show()


if __name__ == '__main__':
    process = gen_data(100)
    output(process)

