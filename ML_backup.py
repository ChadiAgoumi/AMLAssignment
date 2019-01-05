# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Machine Learning - File 1

'-----------------------------------------'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from PIL import Image
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, voting_classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

'-----------------------------------------'

## Import / cleaning

'-----------------------------------------'

# Importing data
df = pd.read_csv('attribute_list.csv', skiprows=1)

# reading pictures. Returning as np.array. Color level: RGB
def read_pic(name, as_black_white=False):   
    string = 'dataset/' + str(name) + '.png'
    if as_black_white == True:
        return(np.array(Image.open(string).convert(mode='L')))
    else:
        return(np.array(Image.open(string)))

# Cleaning data
def clean_data():
    tmp = df[['hair_color', 'eyeglasses', 'smiling', 'young', 'human']]
    tmp['sum'] = tmp.sum(axis=1)
    t = tmp[tmp['sum'] == -5]
    return(t.index + 1)
df.drop(labels=clean_data()-1, inplace=True)

# Converting (-1,1) to (0, 1)
df['eyeglasses'] = (df['eyeglasses'] + 1) / 2
df['smiling'] = (df['smiling'] + 1) / 2
df['young'] = (df['young'] + 1) / 2
df['human'] = (df['human'] + 1) / 2

df['eyeglasses'] = df['eyeglasses'].apply(lambda x: int(x))
df['smiling'] = df['smiling'].apply(lambda x: int(x))
df['young'] = df['young'].apply(lambda x: int(x))
df['human'] = df['human'].apply(lambda x: int(x))

df.head()

'-----------------------------------------'

# Importing Pictures
def import_pictures(data, _as_black_white=False):
    res = []
    file_names = np.array(data['file_name'])
    for k in file_names:
        res.append(read_pic(k, as_black_white=_as_black_white)) # Data are alredy cleaned
    return(np.array(res))

pics_color = import_pictures(df)
pics_color.shape # acces to an image: pics[name_of_img][x_pixel][y_pixel][RGB_color]

'-----------------------------------------'

## First try: black and white

'-----------------------------------------'

### Preprocessing

'-----------------------------------------'

'''Import pictures in __black and white__ and __scaling__.'''

'-----------------------------------------'

pics_b_w = import_pictures(df, _as_black_white=True)
print(pics_b_w.shape)
imgs = np.array([preprocessing.scale(pics_b_w[k]) for k in range(len(pics_b_w))])

'-----------------------------------------'

'''Applying __PCA__.'''

'-----------------------------------------'

# First, ravel
def to_vector(data=pics_b_w):
    res = []
    for k in range(len(pics_b_w)):
        res.append(pics_b_w[k].ravel())
    return(np.array(res))

pics_lin = to_vector()


'-----------------------------------------'

# Apply PCA
s = time.time()
pca = PCA()
pca.fit(pics_lin)

t = time.time()

print('Time to compute pca.fit: ' + str(int(100*(t - s)/60)/100) + ' min')

print('Explained Variance:', pca.explained_variance_ratio_)
# searching for number of component to keep
v_exp = np.cumsum(pca.explained_variance_ratio_)

def extract_var(tbl, x=.95):
    k = 0
    while tbl[k] < x:
        k = k + 1
    return(k)

k95 = extract_var(v_exp)
k99 = extract_var(v_exp, x=.99)

# X = pca.fit_transform(pics_lin)
'''pca95 = PCA(n_components=k95)
X95 = pca95.fit_transform(pics_lin)
print('X95.shape', X95.shape)'''

pca99 = PCA(n_components=k99)
X99 = pca99.fit_transform(pics_lin)
print('X99.shape', X99.shape)
print('Time to compute the two pca.fit_transform (95%, 99%): ' + str(int(100*(time.time() - t)/60)/100) + ' min')

'-----------------------------------------'


### Learning
'-----------------------------------------'

#### Human

'-----------------------------------------'

Y_human = np.array(df['human'])
X_train_h, X_test_h, Y_train_h, Y_test_h = train_test_split(X99, Y_human, test_size=0.2)

# Cross validation score
c1 = np.mean(cross_val_score(MLPClassifier(), X99, Y_human, cv=6))
print('Cross Validation score: ' + str(100*c1) + '%')

# 'By hand'
clf_neuralNetwork_human = MLPClassifier()
clf_neuralNetwork_human.fit(X_train_h, Y_train_h)
print(clf_neuralNetwork_human.score(X_test_h, Y_test_h))

'-----------------------------------------'

# Manual testing (possible because dataset already shuffled)
ratio_train = 0.8
Y = np.array(df['human'])
names = np.array(df['file_name'])
X_train, X_test = X99[:int(ratio_train*len(X99))], X99[int(ratio_train*len(X99)):] 
Y_train, Y_test = Y[:int(ratio_train*len(Y))], Y[int(ratio_train*len(Y)):]
clf_neuralNetwork_mt = MLPClassifier()
clf_neuralNetwork_mt.fit(X_train, Y_train)
names_train, names_test = names[:int(ratio_train*len(X99))], names[int(ratio_train*len(X99)):]
print(clf_neuralNetwork_mt.predict([X_test[20], X_test[21], X_test[22], X_test[23],
                                 X_test[24], X_test[25], X_test[26], X_test[27],
                                 X_test[28], X_test[29], X_test[30], X_test[31]]))
print(names_test[20:32])


