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
    """バイアス項の追加などの前処理"""
    X = df[[2]].values
    y = df[[3]].values

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
    x = np.arange(-0, 8, 0.01)
    best_fit = params[1]*x+params[0]
    ax.set_xlabel('petal_length')
    ax.set_ylabel('petal_width')
    ax.set_xlim([0.0, 8.0])
    ax.set_ylim([0.0, 3.0])
    ax.scatter(X[:, 1],y[:, 0], color='blue', s=5)
    ax.plot(x, best_fit, color='red')
    plt.savefig('result.jpg')
    plt.show()


if __name__ == '__main__':

    df = load_data()
    X, y = preprocess(df)
    params = normal_equation(X, y)
    plot(X, y, params)