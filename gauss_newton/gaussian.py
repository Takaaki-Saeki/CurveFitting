import numpy as np
import matplotlib.pyplot as plt
import sympy as sym


def gaussian1(arr_x, mu, sigma, a):
    return a/(np.sqrt(2*np.pi)*sigma) * np.exp(-(arr_x - mu)**2 / (2*sigma**2))

def generate_data():
    rand = np.random.rand(100)
    rand = (2 * rand - 1) * 0.05
    arr_x = np.arange(100, 200, 1)
    arr_y = gaussian1(arr_x, 150, 10, 30) + rand
    return arr_x, arr_y


def gauss(x, mu, sigma, amp):
    """ミカエリス・メンテンの式"""
    f = amp / (np.sqrt(2*np.pi)*sigma) * sym.exp(-(x - mu)**2 / (2*sigma**2))
    return f


def jacobi_res(p1, p2, p3):
    """ヤコビ行列と残差ベクトルの計算"""
    jacobi = []
    res = []
    for i in range(arr_x.shape[0]):
        f = gauss(arr_x[i], a, b, c)
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


def calc(params):
    """最大反復回数20回として反復計算"""
    n_iteration = 0
    res_list = []
    while n_iteration <= 10:
        J, r = jacobi_res(params[0], params[1], params[2])
        inv = np.linalg.inv(np.dot(J.T, J))
        new_params = params - np.dot(np.dot(inv, J.T), r)
        if np.linalg.norm(new_params - params) < 1.0*10**(-15):
            break
        print(params)
        params = new_params
        res_list.append(np.linalg.norm(r)**2)
        n_iteration += 1
    return params, res_list


def plot(params, res_list):
    """グラフの描画"""
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.5, hspace=0.2)

    x1 = np.arange(100, 200, 1)
    func = gaussian1(x1, params[0], params[1], params[2])
    ax1.set_xlabel('x', fontdict={'fontsize':18})
    ax1.set_ylabel('y', fontdict={'fontsize':18})
    ax1.plot(x1, func, color='red')
    ax1.scatter(arr_x, arr_y, color='blue')

    x2 = np.arange(0, len(res_list), 1)
    ax2.set_yscale('log')
    ax2.set_xlabel('number of iteration', fontdict={'fontsize':18})
    ax2.set_ylabel('S(a)', fontdict={'fontsize':18})
    ax2.plot(x2, np.array(res_list), color='black')
    ax2.set_ylim([1.0*10**(-3), 100.0])
    ax2.set_xlim([0, 20])
    ax2.set_xticks(np.arange(20))

    plt.savefig('gauss-newton_gaussian_fail.jpg')
    plt.show()


if __name__ == '__main__':

    arr_x, arr_y = generate_data()

    (x, a, b, c) = sym.symbols('x a b c')

    init_params = initialize(130, 5, 5)
    params, res_list = calc(init_params)
    plot(params, res_list)