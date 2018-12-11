import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    """irisデータの読み取り"""
    path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/'
    data = 'iris.data'

    df = pd.read_table(path + data, sep=',', header=None)
    df.drop([4], axis=1, inplace=True)
    return df

def preprocess(df):
    """標準化およびバイアス項の追加"""
    X = df[[2]].values
    y = df[[3]].values
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)

    X = np.insert(X, 0, np.ones((1, X.shape[0]), dtype=int), axis=1)

    return X, y


def normal_equation(X, y):
    """正規方程式を解く"""
    A = np.dot(X.T, X)
    B = np.dot(X.T, y)
    a = np.linalg.solve(A, B)
    return a


def plot(X, y, params):
    """結果のプロット"""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.2)
    x = np.arange(-2, 2, 0.01)
    ax.set_xlabel('petal_length (scaled)')
    ax.set_ylabel('petal_width (scaled)')
    ax.scatter(X[:, 1],y[:, 0], color='green')
    ax.plot(X[:, 1], np.dot(X, params)[:, 0], color='red')
    plt.savefig('result.jpg')
    plt.show()


if __name__ == '__main__':

    df = load_data()
    X, y = preprocess(df)
    params = normal_equation(X, y)
    plot(X, y, params)