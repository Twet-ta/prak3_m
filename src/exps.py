import pandas as pd
import re
from sklearn.model_selection import train_test_split
data = pd.read_csv('kc_house_data.csv')
targ = data['price']
data = data.drop(columns='price')
obj_f = []
for i in data.columns:
    if data[i].dtypes == object:
        obj_f.append(i)
print(obj_f)
for i in obj_f:
    data[i] = data[i].apply(lambda x: x.lower())
    data[i] = data[i].apply(lambda x: str(re.sub('\n', ' ', str(x), count=0)))
    data[i] = data[i].apply(lambda x: str(re.sub('[^0-9]', '', str(x), count=0)))
data.loc[:, obj_f] = data[obj_f].astype(int)
data = data.to_numpy()
targ = targ.to_numpy()
D_tr = data[:D_tr.shape[0]*8//10, :]
Y_tr = targ[:D_tr.shape[0]*8//10]
D_vl = data[D_tr.shape[0]*8//10:, :]
Y_vl = targ[D_tr.shape[0]*8//10:]