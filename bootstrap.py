# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 22:52:20 2021

@author: maxim
"""
import time
start_time = time.time()
from sklearn.utils import resample

import numpy as np
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
from keras.models import Sequential 
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import os
import sys

seed=pd.read_csv("seed_list.csv")
seed=seed['# seeds'].tolist()
testnumber=seed[int(sys.argv[1])]

os.chdir('Bootstrap_'+str(int(sys.argv[2])))
os.mkdir('test_bootstrap_'+str(testnumber))
os.chdir('test_bootstrap_'+str(testnumber))

np.random.seed(testnumber)

n_max=20000

n_train=int(sys.argv[3])
if n_train==15000:
    batch_size = 36
elif n_train==100000:
    batch_size = 60
elif n_train==50000:
    batchsize=48
else:
    batchsize=36

def f(x,smoothness):
    return(np.sin(10*x**2))


def g_0(x,smoothness):
    return(np.sin(10*x**2))


def g_1(x,smoothness):
    res = 0
    for i in np.arange(1,np.sqrt(n_max)):
        coef = np.sin(i)/(i**(smoothness+0.5))
        res+=coef*np.cos(np.pi*(i-0.5)*x)
    return(res)

def g_2(x,smoothness):
    res = 0
    for i in np.arange(1,np.sqrt(n_max)):
        coef = np.cos(i**2)/(i**(smoothness+0.5))
        res+=coef*np.cos(np.pi*(i-0.5)*x)
    return(res)

def g_3(x,smoothness):
    res = 0
    for i in np.arange(1,np.sqrt(n_max)):
        coef = (np.sin(i)+np.cos(i))/(i**(smoothness+0.5))
        res+=coef*np.cos(np.pi*(i-0.5)*x)
    return(res)

def g_4(x,smoothness):
    res = 0
    for i in np.arange(1,np.sqrt(n_max)):
        coef = (np.sin(i**2)+np.cos(i**2))/(i**(smoothness+0.5))
        res+=coef*np.cos(np.pi*(i-0.5)*x)
    return(res)


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def generate_data(N,function):
    ylist=[]
    xlist=[]
    for k in range (N):
        x=np.random.uniform(0,1)
        y=np.random.binomial(1,sigmoid(function(x,1)))
        ylist.append(y)
        xlist.append(x)
    ylistsorted=[x for _,x in sorted(zip(xlist,ylist))]
    xlistsorted=sorted(xlist)
    xy=list(zip(xlistsorted,ylistsorted))
    return(xy)



if int(sys.argv[2])==0:
    listsorted=generate_data(n_train,g_0)
    truefunction=[sigmoid(g_0(x,1)) for x in np.linspace(0,1,n_train)]
elif int(sys.argv[2])==1:
    listsorted=generate_data(n_train,g_1)
    truefunction=[sigmoid(g_1(x,1)) for x in np.linspace(0,1,n_train)]
elif int(sys.argv[2])==2:
    listsorted=generate_data(n_train,g_2)
    truefunction=[sigmoid(g_2(x,1)) for x in np.linspace(0,1,n_train)]
elif int(sys.argv[2])==3:
    listsorted=generate_data(n_train,g_3)
    truefunction=[sigmoid(g_3(x,1)) for x in np.linspace(0,1,n_train)]
else:
    listsorted=generate_data(n_train,g_4)
    truefunction=[sigmoid(g_4(x,1)) for x in np.linspace(0,1,n_train)]

listofruns=[]

for runs in range (50):
    y_sample=resample(listsorted, n_samples=n_train)
    

    listsorted_df=pd.DataFrame(y_sample)
    listsorted_df=listsorted_df.sample(frac=1)
    listsorted_df.columns=['x-axis','target']
    
    train_x=np.array(listsorted_df['x-axis'])
    target_x=(listsorted_df['target'])
    test_x=np.linspace(0,1,int(listsorted_df.shape[0]))
    target_x2 = np_utils.to_categorical(np.array(target_x), 2)  
    
    epochs = 100
    
    smoothness=1
    N=train_x.shape[0]
    Depth = int(smoothness * np.floor(np.log(N)))
    Width = int(N**( 1/(1 + 2* smoothness)))
    
    
    model = Sequential()
    model.add(Dense(Width, input_dim=1, activation='relu')) 
    
    for k in range (Depth):
        model.add(Dense(Width, input_dim=Width, activation='relu'))
    
    model.add(Dense(2, input_dim=Width, activation='sigmoid'))
    
    opt = keras.optimizers.Adam()
    #opt = keras.optimizers.SGD()
    #cce = keras.losses.CategoricalCrossentropy()
    cce = keras.losses.BinaryCrossentropy()
    model.compile(loss=cce, metrics=['accuracy'], optimizer=opt)
        
    model.fit(train_x, target_x2, batch_size=batch_size, epochs=epochs, verbose=1,validation_split=0.1)
    
    
    test_x3=np.expand_dims(test_x,1)
    model3=model(test_x3)
    
    reconstructed=[]
    for k in range (len(model3)):
        reconstructed.append([test_x[k],model3[k][1].numpy()])
    
    listofruns.append(reconstructed)





def L2ball(listoffunctions, truefunction):
    
    y_mean=[]

    for k in range (len(listoffunctions[0])):
        y_mean.append( sum([listoffunctions[i][k][1] for i in range (len(listoffunctions))])/len(listoffunctions))

    #plt.plot(test_x,truefunction,color='r')

    L2list=[]
    for k in range (len(listoffunctions)):
        L2list.append(np.sqrt(sum([(listoffunctions[k][i][1]-y_mean[i])**2 for i in range (len(y_mean))]))/len(y_mean))
    
    L2listsorted=sorted(L2list)
    index=int(0.95*len(L2list))
    L2ballradius=L2listsorted[index]

    L2zero=[]
    for k in range (len(y_mean)):
        L2zero.append((y_mean[k]-truefunction[k])**2)
    L2zero=np.sqrt(sum(L2zero))/len(y_mean)
    return(L2ballradius,L2zero)

L2ballradius,L2zero=L2ball(listofruns,truefunction)


x=np.linspace(0,1,len(listofruns[0]))
y_inner_plot=[]
y_outer_plot=[]
y_mean2=[]
for k in range (len(listofruns[0])):
    y_inner_plot.append(min([listofruns[i][k][1] for i in range (len(listofruns))]))
    y_outer_plot.append(max([listofruns[i][k][1] for i in range (len(listofruns))]))
    y_mean2.append( sum([listofruns[i][k][1] for i in range (len(listofruns))])/len(listofruns))

plt.clf()
plt.plot(x,truefunction,'r',zorder=10)
plt.plot(x,y_mean2,'y',zorder=5)
plt.plot(x,y_inner_plot, 'b',zorder=0)
plt.plot(x,y_outer_plot, 'b',zorder=0)
plt.fill_between(x,y_inner_plot,y_outer_plot)
plt.legend(["True function","Mean prediction of all networks","max/min interval"])
fig = plt.gcf()
fig.savefig('traceplots_main_bootstrap.pdf')
plt.clf()

np.savetxt("L2ball_bootstrap.txt",[L2ballradius,L2zero])

t=(time.time() - start_time)
np.savetxt("time.txt",[t])

    
