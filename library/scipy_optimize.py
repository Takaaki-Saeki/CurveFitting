import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def gaussian(arr_x, mean, sigma, a):
    """ガウス関数 (引数に値をとる)"""
    return a/(np.sqrt(2*np.pi)*sigma) * np.exp(-(arr_x - mean)**2 / (2*sigma**2))


def generate_data():
    """フィッティング用データの生成"""
    rand = np.random.rand(100)
    rand = (2 * rand - 1) * 0.05
    arr_x = np.arange(100, 200, 1)
    arr_y = gaussian(arr_x, 150, 10, 30) + rand
    return arr_x, arr_y


def initialize(init_a, init_b, init_c):
    """パラメータベクトルの初期化"""
    params = np.array([init_a, init_b, init_c])
    return params


def plot(params):
    """グラフの描画"""
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.2)

    x1 = np.arange(100, 200, 1)
    func = gaussian(x1, params[0], params[1], params[2])
    ax.set_xlabel('x', fontdict={'fontsize':18})
    ax.set_ylabel('y', fontdict={'fontsize':18})
    ax.plot(x1, func, color='red')
    ax.scatter(arr_x, arr_y, color='blue', s=5)

    plt.savefig('scipy_optimize.jpg')
    plt.show()


if __name__ == '__main__':

    arr_x, arr_y = generate_data()
    init_params = initialize(150, 5, 5)
    opt_params, cov = scipy.optimize.curve_fit(gaussian, arr_x, arr_y, p0=init_params)
    plot(opt_params)
    print(cov)
