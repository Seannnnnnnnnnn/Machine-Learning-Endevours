"""
A python file for generating plots and data of a random variable
"""
import random
from typing import List
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def gen_data(n) -> List:
    m, c = random.random(), random.random()
    y = [m*x+c + random.gauss(0, 2) for x in range(n)]
    return y


def gen_data_frame(n):
    return pd.DataFrame(
        {
            'x': [np.random.exponential(2) for _ in range(n)],
            'y': [random.gauss(0, 2) for _ in range(n)]
        }
    )


if __name__ == '__main__':
    process = gen_data(100)


