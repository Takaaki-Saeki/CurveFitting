import numpy as np
import matplotlib.pyplot as plt
import sympy as sym


def gaussian(arr_x, mean, sigma, a):
    """ガウス関数 (引数に値をとる)"""
    return a/(np.sqrt(2*np.pi)*sigma) * np.exp(-(arr_x - mean)**2 / (2*sigma**2))


def fitting_func(x, mean, sigma, amp):
    """ガウス関数 (引数にsympyのsumbolを取る)"""
    f = amp / (np.sqrt(2*np.pi)*sigma) * sym.exp(-(x - mean)**2 / (2*sigma**2))
    return f


def generate_data():
    """フィッティング用データの生成"""
    rand = np.random.rand(100)
    rand = (2 * rand - 1) * 0.05
    arr_x = np.arange(100, 200, 1)
    arr_y = gaussian(arr_x, 150, 10, 30) + rand
    return arr_x, arr_y


def jacobi_res(p1, p2, p3):
    """ヤコビ行列と残差ベクトルの計算"""
    jacobi = []
    res = []
    for i in range(arr_x.shape[0]):
        f = fitting_func(arr_x[i], a, b, c)
        r = arr_y[i] - f
        j1 = sym.diff(r, a)
        j2 = sym.diff(r, b)
        j3 = sym.diff(r, c)
        j1 = j1.subs([(a, p1), (b, p2), (c, p3)])
        j2 = j2.subs([(a, p1), (b, p2), (c, p3)])
        j3 = j3.subs([(a, p1), (b, p2), (c, p3)])
        j = [j1, j2, j3]
        r = r.subs([(a, p1), (b, p2), (c, p3)])
        jacobi.append(j)
        res.append(r)
    jacobi = np.array(jacobi).astype('float')
    res = np.array(res).astype('float')
    return jacobi, res


def initialize(init_a, init_b, init_c):
    """パラメータベクトルの初期化"""
    params = np.array([init_a, init_b, init_c])
    return params


def levenberg(params):
    """最大反復回数100回として反復計算"""
    n_iteration = 0
    res2_list = []
    lam = 0.01
    alpha = 1.0
    res_old = np.inf
    threshold = 1000
    init_params = params
    while n_iteration <= 100:
        J, r = jacobi_res(params[0], params[1], params[2])
        JJ = np.dot(J.T, J)
        inv = np.linalg.inv(JJ + lam*np.diag(np.diag(JJ)))
        new_params = params - alpha*np.dot(np.dot(inv, J.T), r)
        determination = np.linalg.norm(new_params - params)/np.linalg.norm(params)
        if determination < 1.0*10**(-5):
            break
        if np.linalg.norm(r) < res_old:
            lam = lam/10
        else:
            lam = lam*10
        params = new_params
        res_old = np.linalg.norm(r)
        res2_list.append(np.linalg.norm(r)**2)
        n_iteration += 1
        if abs(params[0]) > threshold or abs(params[1]) > threshold or abs(params[2]) > threshold:
            alpha = alpha / 10
            n_iteration = 0
            params = init_params
            res2_list = []
    return params, res2_list


def plot(params, res2_list):
    """グラフの描画"""
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.5, hspace=0.2)

    x1 = np.arange(100, 200, 1)
    func = gaussian(x1, params[0], params[1], params[2])
    ax1.set_xlabel('x', fontdict={'fontsize':18})
    ax1.set_ylabel('y', fontdict={'fontsize':18})
    ax1.plot(x1, func, color='red')
    ax1.scatter(arr_x, arr_y, color='blue')

    x2 = np.arange(0, len(res2_list), 1)
    ax2.set_yscale('log')
    ax2.set_xlabel('number of iteration', fontdict={'fontsize':18})
    ax2.set_ylabel('S(a)', fontdict={'fontsize':18})
    ax2.plot(x2, np.array(res2_list), color='black')
    ax2.set_ylim([1.0*10**(-3), 100.0])
    ax2.set_xlim([0, 100])

    plt.savefig('levenberg.jpg')
    plt.show()


if __name__ == '__main__':

    arr_x, arr_y = generate_data()

    (x, a, b, c) = sym.symbols('x a b c')

    init_params = initialize(130, 5, 5)
    params, res2_list = levenberg(init_params)
    plot(params, res2_list)










