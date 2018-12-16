import numpy as np
import matplotlib.pyplot as plt
import sympy as sym


def michaelis(x, a, b):
    """ミカエリス・メンテンの式"""
    f = a*x / (x + b)
    return f


def jacobi_res(p1, p2):
    """ヤコビ行列と残差ベクトルの計算"""
    jacobi = []
    res = []
    for i in range(arr_S.shape[0]):
        f = michaelis(arr_S[i], a, b)
        r = arr_v[i] - f
        j1 = sym.diff(r, a)
        j2 = sym.diff(r, b)
        j1 = j1.subs([(a, p1), (b, p2)])
        j2 = j2.subs([(a, p1), (b, p2)])
        j = [j1, j2]
        r = r.subs([(a, p1), (b, p2)])
        jacobi.append(j)
        res.append(r)
    jacobi = np.array(jacobi).astype('float')
    res = np.array(res).astype('float')
    return jacobi, res


def initialize(init_a, init_b):
    """パラメータベクトルの初期化"""
    params = np.array([init_a, init_b])
    return params


def gauss_newton(params):
    """最大反復回数20回として反復計算"""
    n_iteration = 0
    res_list = []
    while n_iteration <= 20:
        J, r = jacobi_res(params[0], params[1])
        inv = np.linalg.inv(np.dot(J.T, J))
        new_params = params - np.dot(np.dot(inv, J.T), r)
        if np.linalg.norm(new_params - params)/np.linalg.norm(params) < 1.0*10**(-15):
            break
        params = new_params
        res_list.append(np.linalg.norm(r)**2)
        n_iteration += 1
    return params, res_list


def plot(params, res_list):
    """グラフの描画"""
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.5, hspace=0.2)

    x1 = np.arange(0, 4, 0.01)
    func = params[0] * x1 / (x1 + params[1])
    ax1.set_xlabel('[s]', fontdict={'fontsize':18})
    ax1.set_ylabel('v', fontdict={'fontsize':18})
    ax1.plot(x1, func, color='red')
    ax1.scatter(arr_S, arr_v, color='blue')

    x2 = np.arange(0, len(res_list), 1)
    ax2.set_yscale('log')
    ax2.set_xlabel('number of iteration', fontdict={'fontsize':18})
    ax2.set_ylabel('S(a)', fontdict={'fontsize':18})
    ax2.plot(x2, np.array(res_list), color='black')
    ax2.set_ylim([1.0*10**(-3), 1.0])
    ax2.set_xlim([0, 20])
    ax2.set_xticks(np.arange(20))

    plt.savefig('gauss-newton.jpg')
    plt.show()


if __name__ == '__main__':

    arr_S = np.array([0.038, 0.194, 0.425, 0.626, 1.253, 2.500, 3.740])
    arr_v = np.array([0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317])

    (x, a, b) = sym.symbols('x a b')

    init_params = initialize(0.9, 0.2)
    params, res_list = gauss_newton(init_params)
    plot(params, res_list)
