import pandas as pd
import matplotlib.pyplot as plt

"""データの読み取り"""
data = pd.read_csv('gauss_data.csv', header=None)

x = data[1]
y = data[0]

"""フィッティングしたいモデルのインポート"""
from lmfit.models import ExponentialModel, GaussianModel

"""expモデルの定義"""
# パラメータオブジェクトparsの生成
exp_mod = ExponentialModel(prefix='exp_')
pars = exp_mod.guess(y, x=x)

# 1つ目のピーク
gauss1 = GaussianModel(prefix='g1_')
pars.update(gauss1.make_params())
pars['g1_center'].set(55, min=40, max=100)
pars['g1_sigma'].set(10, min=3)
pars['g1_amplitude'].set(2000, min=10)

# 2つ目のピーク
gauss2 = GaussianModel(prefix='g2_')
pars.update(gauss2.make_params())
pars['g2_center'].set(170, min=150, max=200)
pars['g2_sigma'].set(10, min=3)
pars['g2_amplitude'].set(2000, min=10)

# モデルの合成
mod = gauss1 + gauss2 + exp_mod

# 初期値
init = mod.eval(pars, x=x)

# 最適値
out = mod.fit(y, pars, x=x)

fig, ax = plt.subplots(figsize=(4, 3))
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(left=0.2)

ax.scatter(x, y, s=5, color='blue')
ax.plot(x, out.best_fit, 'r-')
ax.set_xlabel('wavelength (nm)')
ax.set_ylabel('intensity (a.u.)')

plt.savefig('lm.jpg')
plt.show()