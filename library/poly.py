import numpy as np
import matplotlib.pyplot as plt

def pol2(x, a, b, c):
    return a*x**2 + b*x + c


def generate_data():
    rand = np.random.rand(100)
    rand = (2 * rand - 1) * 10
    x = np.arange(0, 10, 0.1)
    y = pol2(x, 2, 1, 0.1) + rand
    return x, y


def plot(params):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.5, hspace=0.2)

    ax1.scatter(x, y, color='red')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    x1 = np.arange(0, 10, 0.01)
    func = pol2(x1, params[0], params[1], params[2])
    ax2.plot(x1, func, color='black')
    ax2.scatter(x, y, color='red')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.savefig('poly.jpg')
    plt.show()


if __name__ == '__main__':
    x, y = generate_data()
    params = np.polyfit(x, y, 2)
    plot(params)


