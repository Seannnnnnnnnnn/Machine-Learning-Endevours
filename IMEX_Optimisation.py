""" contains code for IMEX optimisation assignment """
import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def get_data():
    with open('Data.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        x, y, q = [], [], []
        for row in reader:
            x.append(int(row[1]))
            y.append(int(row[2]))
            q.append(int(row[3]))
    return pd.DataFrame({'x': x, 'y': y, 'q': q})


def k_means(k=5):
    """ implements the k means algorithm """

    data = get_data()

    # define our inital allocation:
    centroids = {
                i+1: [np.random.randint(5, 140), np.random.randint(6, 122)] for i in range(k)
                }

    df = assignment(data, centroids)

    def update(k):
        """ Completes an update iteration of the algorithm """
        for i in centroids.keys():
            centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
            centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
        return k

    # now update until convergance:
    while True:
        closest = df['closest'].copy(deep=True)
        centroids = update(centroids)
        df = assignment(df, centroids)
        if closest.equals(df['closest']):
            break

    print('Locations: ')
    print(centroids)

    plt.scatter(data['x'], data['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i])
    plt.xlim(0, 140)
    plt.ylim(0, 140)
    plt.show()
    return


def assignment(data_frame, centroid):
    """ Completes an assignment of the k-means iteration """
    for i in centroid.keys():
        data_frame['distance_from_{}'.format(i)] = \
            (
             data_frame['q'] * abs((centroid[i][0]) - data_frame['x'])) + abs((centroid[i][1]) - data_frame['y']
            )
    distance_col = ['distance_from_{}'.format(i) for i in centroid.keys()]
    data_frame['closest'] = data_frame.loc[:, distance_col].idxmin(axis=1)
    data_frame['closest'] = data_frame['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    data_frame['color'] = data_frame['closest'].map(lambda x: colmap[x])
    return data_frame


if __name__ == '__main__':
    colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'grey', 5: 'pink'}
    k_means(5)
