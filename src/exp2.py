import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import timeit
from scipy.optimize import minimize_scalar
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import random
import matplotlib
import matplotlib.pyplot as plt

import scipy
import matplotlib.ticker
import seaborn as sns
import pickle

gb_list_n = []
rf_list_n = []
n = 1000
Cls = RandomForestMSE(n, max_depth=None, random_state=0)
Cl = GradientBoostingMSE(n, max_depth=None, random_state=0, learning_rate=0.1)
gb_list_n = Cl.fit(D_tr, Y_tr, D_vl, Y_vl)
rf_list_n = Cls.fit(D_tr, Y_tr, D_vl, Y_vl)

x1 = np.array(np.arange(len(rf_list_n['acc'])), dtype=int)
plt.plot(x1[1:], rf_list_n['acc'][1:])
plt.xlabel('Количество деревьев')
plt.ylabel('RMSE')
plt.title('Зависимость RMSE от количества деревьев, RF', fontsize=10)
plt.grid(True)
plt.legend()
plt.savefig("rn_rf.pdf", format="pdf", bbox_inches='tight')
plt.show()

x1 = np.array(np.arange(len(rf_list_n['time'])), dtype=int)
plt.plot(x1[1:], rf_list_n['time'][1:])
#plt.plot(sgd_list[i]['epoch_num'][1:], sgd_list[i]['accuracy'][1:], color=((i%3)*0.3, (i)*0.1, (i)*0.1), label=['sgd=', (i+1)*0.5])
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Количество деревьев')
plt.ylabel('Время, с')
plt.title('Зависимость времени от количества деревьев, RF', fontsize=10)
plt.grid(True)
plt.legend()
#plt.yticks([i*0.05 for i in range(8, 18)])
#plt.xscale('log')
plt.savefig("tn_rf.pdf", format="pdf", bbox_inches='tight')
plt.show()

