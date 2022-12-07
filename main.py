import time
start_time = time.time()
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
import theano
import pymc3 as pm
import sys

seed=pd.read_csv("seed_list.csv")
seed=seed['# seeds'].tolist()
testnumber=seed[int(sys.argv[1])]
os.chdir('EBNN_'+str(int(sys.argv[2])))
os.mkdir('test_'+str(testnumber))
os.chdir('test_'+str(testnumber))

np.random.seed(testnumber)
#Definitions of function. n_max is the resolution of our functions. n_train is the number of training points.

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


# np.savetxt("truefunction_ALICE_g_1.csv", truefunction,delimiter=",") 
# np.savetxt("5000points_test_ALICE_g_1.csv", listsorted, delimiter=",")

# We transfer the data into a Pandas Dataframe
listsorted_df=pd.DataFrame(listsorted)
listsorted_df=listsorted_df.sample(frac=1)
listsorted_df.columns=['x-axis','target']
data=listsorted_df[int(listsorted_df.shape[0]/2):]
data2=listsorted_df[:int(listsorted_df.shape[0]/2)]
#data=listsorted_df[0:]
#data2=listsorted_df[0:]
#Neural Network
from keras.models import Sequential 
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import keras
#np.random.seed(100)

train_x=np.array(data['x-axis'])
target_x=(data['target'])
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



# test_x3=np.expand_dims(test_x,1)
# model3=model(test_x3)
# reconstructed=[]
# for k in range (len(model3)):
#     reconstructed.append([test_x[k],model3[k][1].numpy()])
# x_sigmoid=[x[0] for x in reconstructed ]
# y_sigmoid=[y[1] for y in reconstructed ]

# plt.figure(figsize=(8.0, 4.5))
# plt.plot(np.linspace(0,1,len(truefunction)),truefunction,label="True function")
# plt.plot(x_sigmoid,y_sigmoid,label="sigmoid/BCE epochs/batch=100/32")
# plt.legend()
# plt.legend(loc='upper right')
#plt.savefig('test.png')

# Save the output of the last hidden layer
from keras.models import Model

data2=data2.sort_values(by='x-axis')
layers_names=[layer.name for layer in model.layers ]
layer_name = layers_names[-2]
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
train_x2=np.array(data2['x-axis'])
intermediate_output = intermediate_layer_model.predict(train_x2)

# neurons=[]
# for k in range (intermediate_output.shape[1]):
#     neurons.append([intermediate_output[i][k] for i in range (intermediate_output.shape[0])])

dataframe_output_by_axis=pd.DataFrame(intermediate_output)
dataframe_output_by_axis['x-axis']=data2['x-axis'].values
dataframe_output_by_axis['target']=data2['target'].values

#np.savetxt("network_last_layer_output.csv", dataframe_output_by_axis, delimiter=",")


for k in range (dataframe_output_by_axis.shape[1]):
    dataframe_output_by_axis.rename(columns={k:'weight_'+str(k)}, inplace=True)

del dataframe_output_by_axis['x-axis']



# RANDOM_SEED = 2000
# np.random.seed(RANDOM_SEED)

# def plot_traces(traces, model, retain=0):
#     """
#     Convenience function:
#     Plot traces with overlaid means and values
#     """
#     with model:
#         ax = pm.traceplot(
#             traces[-retain:],
#             lines=tuple([(k, {}, v["mean"]) for k, v in pm.summary(traces[-retain:]).iterrows()]),
#         )

#         for i, mn in enumerate(pm.summary(traces[-retain:])["mean"]):
#             ax[i, 0].annotate(
#                 f"{mn:.2f}",
#                 xy=(mn, 0),
#                 xycoords="data",
#                 xytext=(5, 10),
#                 textcoords="offset points",
#                 rotation=90,
#                 va="bottom",
#                 fontsize="large",
#                 color="#AA0022",
#             )

# Remove neurons with empty outputs
listenonzeroweights=[]
for k in range (dataframe_output_by_axis.shape[1]-1):
    if (dataframe_output_by_axis['weight_'+str(k)]==0).all()==False:
        listenonzeroweights.append(k)

string_weight_list=['weight_'+str(listenonzeroweights[0])]
string_weight='weight_'+str(listenonzeroweights[0])
for k in listenonzeroweights[1:]:
    string_weight+=' + '+'weight_'+str(k)
    string_weight_list.append('weight_'+str(k))

    
start_time_regression = time.time()
# Perform Bayesian Linear Regression
if __name__ ==  '__main__':
    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula(
            str("target ~ "+string_weight), dataframe_output_by_axis, family=pm.glm.families.Binomial()
        )
        trace = pm.sample(200, tune=1000, init="adapt_diag", cores=1, step=pm.Metropolis())


#plot_traces(trace, logistic_model)
#fig = plt.gcf()
#fig.savefig('trace_plot_ALICE.png')
#plt.clf()


intercept=[]
tracedata=[]
for k in range (len(trace)):
    intercept.append(trace[k]['Intercept'])
    tracedata.append([trace[k]['weight_'+str(i)] for i in listenonzeroweights])


#Store the trace in a pandas dataframe
np.savetxt("trace_ALICE.csv", tracedata, delimiter=",", header=str([k for k in string_weight_list])[1:len(str(string_weight_list))-1],comments='')
np.savetxt("trace_ALICE_intercept.csv", intercept, delimiter=",")

tracedata=pd.read_csv("trace_ALICE.csv")
intercept=pd.read_csv("trace_ALICE_intercept.csv", header= None)


# import os
# if os.path.exists("trace_ALICE.csv"):
#   os.remove("trace_ALICE.csv")
# else:
#   print("The file does not exist") 
  
  
# if os.path.exists("trace_ALICE_intercept.csv"):
#   os.remove("trace_ALICE_intercept.csv")
# else:
#   print("The file does not exist") 
  
interceptlist=intercept.values.tolist()
original_length=dataframe_output_by_axis.shape[1]-1

#Restore original data shape
for k in range (original_length):
    if k not in listenonzeroweights:
        tracedata.insert(k,'weight_'+str(k),0)
del dataframe_output_by_axis['target']

tracelist=tracedata.values.tolist()

y_mega_list_of_plots=[]

#Perform dot product of matrices
for k in range (len(tracedata)):
    y_mega_list_of_plots.append([sum([x*y for x,y in zip(dataframe_output_by_axis.iloc[i].tolist(),tracelist[k])])+interceptlist[k][0] for i in range (dataframe_output_by_axis.shape[0])]) 
    
#np.savetxt("megalistofplots_ALICE.csv",y_mega_list_of_plots,delimiter=",")

for k in range (len(y_mega_list_of_plots)):
    for i in range (len(y_mega_list_of_plots[k])):
        y_mega_list_of_plots[k][i]=sigmoid(y_mega_list_of_plots[k][i])


x=np.linspace(0,1,len(y_mega_list_of_plots[0]))
# for u in range (len(y_mega_list_of_plots)):
#     plt.plot(x,y_mega_list_of_plots[u])

#fig = plt.gcf()
#fig.savefig('traceplots.png')
#plt.clf()

#Calculate the mean of all plots
y_mean=[]
for k in range (len(y_mega_list_of_plots[0])):
    y_mean.append( sum([y_mega_list_of_plots[i][k] for i in range (len(y_mega_list_of_plots))])/len(y_mega_list_of_plots))
    
# plt.plot(x,y_mean)

# fig = plt.gcf()
# fig.savefig('traceplots_mean.png')
# plt.clf()

y_inner_plot=[]
y_outer_plot=[]
for k in range (len(y_mega_list_of_plots[0])):
    y_inner_plot.append(min([y_mega_list_of_plots[i][k] for i in range (len(y_mega_list_of_plots))]))
    y_outer_plot.append(max([y_mega_list_of_plots[i][k] for i in range (len(y_mega_list_of_plots))]))

truefunction_halve=truefunction[1::2]
xtrue=np.linspace(0,1,len(truefunction_halve))


plt.plot(xtrue,truefunction_halve,'r',zorder=10)
#xtrue=np.linspace(0,1,len(truefunction))
#plt.plot(xtrue,truefunction,'r',zorder=10)

plt.plot(x,y_mean,'y',zorder=5)
plt.plot(x,y_inner_plot, 'b',zorder=0)
plt.plot(x,y_outer_plot, 'b',zorder=0)
plt.fill_between(x,y_inner_plot,y_outer_plot)
plt.legend(["True function","Credible mean","max/min interval"])

fig = plt.gcf()
fig.savefig('traceplots_main.pdf')
plt.clf()


#Calculate the L2 ball radius
L2list=[]
for k in range (len(y_mega_list_of_plots)):
    L2list.append(np.sqrt(sum([(y_mega_list_of_plots[k][i]-y_mean[i])**2 for i in range (len(y_mean))]))/len(y_mean))

L2listsorted=sorted(L2list)
index=int(0.95*len(L2list))
L2ballradius=L2listsorted[index]


L2zero=[]
for k in range (len(y_mean)):
    L2zero.append((y_mean[k]-truefunction_halve[k])**2)

#for k in range (len(y_mean)):
#    L2zero.append((y_mean[k]-truefunction[k])**2)
    
L2zero=np.sqrt(sum(L2zero))/len(y_mean)



np.savetxt("L2ball.txt",[L2ballradius,L2zero])
t_2=(time.time() - start_time_regression)
t=(time.time() - start_time)
np.savetxt("time_regression.txt",[t_2])
np.savetxt("time.txt",[t])
