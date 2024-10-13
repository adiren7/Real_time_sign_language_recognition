import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])


def pad_or_trim(lst, target_length):
    return lst[:target_length] + [0]*(target_length - len(lst))

target_length = 42

#convert data format to df format
data_uniform = np.array([pad_or_trim(lst, target_length) for lst in data])

df = pd.DataFrame(data_uniform)

#add target column
df["target"]= labels
df["target"] = pd.to_numeric(df["target"])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)


f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

