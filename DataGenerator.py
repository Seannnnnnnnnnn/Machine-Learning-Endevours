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


def gen_data_frame(n) -> pd.DataFrame:
    """
    a function for generating data to perform linear regression 
    """
    return pd.DataFrame(
        {
            'x': [np.random.exponential(2) for _ in range(n)],
            'y': [random.gauss(0, 2) for _ in range(n)]
        }
    )


def gen_logistic_data(n, m) -> pd.DataFrame:
    """
    a function for generating data to perform logistic regression 
    param: n: size of cluster 1 
    param: m: size of cluster 2
    """
    df = pd.DataFrame(
        {
            "x": np.append([random.uniform(0, 1) for _ in range(n)], [random.uniform(0.75, 1.5) for _ in range(m)]),
            "y": np.append([random.uniform(0, 1) for _ in range(n)], [random.uniform(0.75, 1.5) for _ in range(m)])
        }
    )

    return df


def gen_logistic_data_vis(n, m):
    """
    A function for visualising generated logistic data 
    """
    df = pd.DataFrame(
        {
            "x1": [random.uniform(0, 1) for _ in range(n)],
            "y1": [random.uniform(0, 1) for _ in range(n)],
            "x2": [random.uniform(0.75, 1.5) for _ in range(m)],
            "y2": [random.uniform(0.75, 1.5) for _ in range(m)],
        }
    )
    
    plt.scatter(df['x1'], df['y1'])
    plt.scatter(df['x2'], df['y2'])
    plt.show()
   
    return 

if __name__ == '__main__':
    process = gen_data(100)


