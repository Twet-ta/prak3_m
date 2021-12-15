gb_list_n = []
rf_list_n = []
n = 1000
Cls = RandomForestMSE(n, max_depth=None, random_state=0)
Cl = GradientBoostingMSE(n, max_depth=None, random_state=0, learning_rate=0.1)
gb_list_n = Cl.fit(D_tr, Y_tr, D_vl, Y_vl)
rf_list_n = Cls.fit(D_tr, Y_tr, D_vl, Y_vl)

x1 = np.array(np.arange(len(gb_list_n['acc'])), dtype=int)
plt.plot(x1[1:], gb_list_n['acc'][1:])
#plt.plot(sgd_list[i]['epoch_num'][1:], sgd_list[i]['accuracy'][1:], color=((i%3)*0.3, (i)*0.1, (i)*0.1), label=['sgd=', (i+1)*0.5])
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Количество деревьев')
plt.ylabel('RMSE')
plt.title('Зависимость RMSE от количества деревьев, GB', fontsize=10)
plt.grid(True)
plt.legend()
#plt.yticks([i*0.05 for i in range(8, 18)])
plt.xscale('log')
plt.savefig("rn_gb.pdf", format="pdf", bbox_inches='tight')
plt.show()

x1 = np.array(np.arange(len(gb_list_n['time'])), dtype=int)
plt.plot(x1[1:], gb_list_n['time'][1:])
#plt.plot(sgd_list[i]['epoch_num'][1:], sgd_list[i]['accuracy'][1:], color=((i%3)*0.3, (i)*0.1, (i)*0.1), label=['sgd=', (i+1)*0.5])
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Количество деревьев')
plt.ylabel('Время, с')
plt.title('Зависимость времени от количества деревьев, GB', fontsize=10)
plt.grid(True)
plt.legend()

#plt.yticks([i*0.05 for i in range(8, 18)])
#plt.xscale('log')
plt.savefig("tn_gb.pdf", format="pdf", bbox_inches='tight')
plt.show()

gb_list_s = []
rf_list_s = []
s = D_tr.shape[1]
n = 500
for i in range(1, 11):
    Cls = RandomForestMSE(1000, max_depth=None, random_state=0, feature_subsample_size=s*i//10)
    Cl = GradientBoostingMSE(300, max_depth=None, random_state=0, feature_subsample_size=s*i//10, learning_rate=0.1)
    gb_list_s.append(Cl.fit(D_tr, Y_tr, D_vl, Y_vl))
    rf_list_s.append(Cls.fit(D_tr, Y_tr, D_vl, Y_vl))

x1 = [i*0.1 for i in range(1, 11)]
y1 = []
for i in range(1, 11):
    y1.append(gb_list_s[i - 1]['acc'][-1])
plt.plot(x1, y1, marker='.')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('feature subsample size, part of number of objects')
plt.ylabel('RMSE')
plt.title('Зависимость RMSE от feature subsample size, GB', fontsize=10)
plt.grid(True)
plt.legend()
plt.xticks(x1)
#plt.yticks([i*0.05 for i in range(8, 18)])
#plt.xscale('log')
plt.savefig("rs_gb.pdf", format="pdf", bbox_inches='tight')
plt.show()

x1 = [i*0.1 for i in range(1, 11)]
y1 = []
for i in range(1, 11):
    y1.append(gb_list_s[i - 1]['time'][-1])
plt.plot(x1, y1, marker='.')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('feature subsample size, part of number of objects')
plt.ylabel('Время, с')
plt.title('Зависимость времени от feature subsample size, GB', fontsize=10)
plt.grid(True)
plt.legend()
plt.xticks(x1)
#plt.yticks([i*0.05 for i in range(8, 18)])
#plt.xscale('log')
plt.savefig("ts_gb.pdf", format="pdf", bbox_inches='tight')
plt.show()

gb_list_d = []
rf_list_d = []
s = D_tr.shape[1]
for d in range(1, 11):
    Cls = RandomForestMSE(1000, max_depth=d, random_state=0, feature_subsample_size=s)
    Cl = GradientBoostingMSE(300, max_depth=d, random_state=0, feature_subsample_size=s, learning_rate=0.1)
    gb_list_s.append(Cl.fit(D_tr, Y_tr, D_vl, Y_vl))
    rf_list_s.append(Cls.fit(D_tr, Y_tr, D_vl, Y_vl))
d = None
Cls = RandomForestMSE(n, max_depth=d, random_state=0, feature_subsample_size=s)
Cl = GradientBoostingMSE(n, max_depth=d, random_state=0, feature_subsample_size=s, learning_rate=0.1)
gb_list_s.append(Cl.fit(D_tr, Y_tr, D_vl, Y_vl))
rf_list_s.append(Cls.fit(D_tr, Y_tr, D_vl, Y_vl))

x1 = [i for i in range(0, 11)]
y1 = []
for i in range(1, 12):
    y1.append(gb_list_s[i + 10]['acc'][-1])
plt.plot(x1, y1, marker='.')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Глубина')
plt.ylabel('RMSE')
plt.title('Зависимость RMSE от глубины, GB', fontsize=10)
plt.grid(True)
plt.legend()
x1 = [str(i) for i in range(0, 11)]
x1[0] = "None"
plt.xticks(range(0, 11), x1)
#plt.yticks([i*0.05 for i in range(8, 18)])
#plt.xscale('log')
plt.savefig("rd_gb.pdf", format="pdf", bbox_inches='tight')
plt.show()

x1 = [i for i in range(0, 11)]
y1 = []
for i in range(1, 12):
    y1.append(gb_list_s[i + 10]['time'][-1])
plt.plot(x1, y1, marker='.')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Глубина')
plt.ylabel('Время, с')
x1 = [str(i) for i in range(0, 11)]
x1[0] = "None"
plt.xticks(range(0, 11), x1)
plt.title('Зависимость времени от глубины, GB', fontsize=10)
plt.grid(True)
plt.legend()
#plt.yticks([i*0.05 for i in range(8, 18)])
#plt.xscale('log')
plt.savefig("td_gb.pdf", format="pdf", bbox_inches='tight')
plt.show()

gb_list_lr = []
s = D_tr.shape[1]
for i in range(1, 11):
    Cl = GradientBoostingMSE(300, max_depth=6, random_state=0, feature_subsample_size=s, learning_rate=i*0.1)
    gb_list_lr.append(Cl.fit(D_tr, Y_tr, D_vl, Y_vl))

x1 = [i*0.1 for i in range(1, 11)]
y1 = []
for i in range(1, 11):
    y1.append(gb_list_s[i - 1]['acc'][-1])
plt.plot(x1, y1, marker='.')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Learning rate')
plt.ylabel('RMSE')
plt.title('Зависимость RMSE от темпа обучения, GB', fontsize=10)
plt.grid(True)
plt.legend()

plt.xticks(x1)
#plt.yticks([i*0.05 for i in range(8, 18)])
#plt.xscale('log')
plt.savefig("rlr_gb.pdf", format="pdf", bbox_inches='tight')
plt.show()

x1 = [i*0.1 for i in range(1, 11)]
y1 = []
for i in range(1, 11):
    y1.append(gb_list_s[i - 1]['time'][-1])
plt.plot(x1, y1, marker='.')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Learning rate')
plt.ylabel('Время, с')
plt.xticks(x1)
plt.title('Зависимость времени от темпа обучения, GB', fontsize=10)
plt.grid(True)
plt.legend()
#plt.yticks([i*0.05 for i in range(8, 18)])
#plt.xscale('log')
plt.savefig("tlr_gb.pdf", format="pdf", bbox_inches='tight')
plt.show()