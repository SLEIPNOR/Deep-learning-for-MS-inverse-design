import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
def gaussian_distibution(center=50, sigma=1, length = 100):
    data = np.array(list(range(length)))
    data = np.exp((-(data-center)*(data-center))/(2*sigma*sigma)) / (sigma*math.sqrt(2*math.pi))
    return data.reshape(1, length)

#%% TF Generation
# 3125
Spec= np.array(pd.read_csv('training_T.csv', header=None))
#%%
N = 31250
for i in range(N,N+5000):

    # data = pd.read_table(r"D:\Study\MRes study material\2nd project\fdtd\training_data\results_1\results_{0}.txt".format(i), sep='\t',
    #                      header=None)
    data = Spec[:,i]
    data = np.flip(np.array(data), 0)
    # plt.figure(figsize=(3, 3))
    #
    # plt.plot( data)
    #
    # plt.show()


    plt.figure(figsize=(4, 4))
    plt.specgram(data,xextent = (400,800))
    #
    plt.axis('off')
    plt.savefig('TF\TF_{0}'.format(i),bbox_inches='tight',pad_inches=0.0)

    # plt.show()

    print('\r'"TF Generation Process:{0}%".format(round((i + 1) * 100 /N,3)), end="",
          flush=True)
#%% Spec

N = 1500
data_M = np.array([[]]*512)
for i in range(N):

    data = pd.read_table(r"D:\Study\MRes study material\2nd project\fdtd\training_data\results_test\results_{0}.txt".format(i), sep='\t',
                         header=None)
    data = np.array(data)[:,1].reshape(512,1)
    data_M = np.append(data_M, data, axis=1)

dataframe = pd.DataFrame(data_M)
dataframe.to_csv("testing.csv",index=False,header=False,sep=',')

#%% Geo
N = 1
data_M = np.array([[]]*10)
data_M = data_M.reshape(0,10)
for i in range(N):
    data = pd.read_table(r"D:\Study\MRes study material\2nd project\fdtd\training_data\results_test\GeoPara_{0}.txt".format(i),
                         sep='\t',
                         header=None)
    data = np.array(data)
    data = data[:,1:]
    data_M = np.append(data_M, data, axis=0)

dataframe = pd.DataFrame(data_M)
dataframe.to_csv("Geo_test.csv",index=False,header=False,sep=',')

#%%

Spec= np.array(pd.read_csv('training_T.csv', header=None))
Spec_1= np.array(pd.read_csv('training_4.csv', header=None))
data_M = np.append(Spec, Spec_1, axis=1)
dataframe = pd.DataFrame(data_M)
dataframe.to_csv("training_T.csv",index=False,header=False,sep=',')
#%%
Geo = np.array(pd.read_csv('Geo_T.csv', header=None))
Geo_1 = np.array(pd.read_csv('Geo_4.csv', header=None))
Geo_T = np.append(Geo, Geo_1, axis=0)
dataframe = pd.DataFrame(Geo_T)
dataframe.to_csv("Geo_T.csv",index=False,header=False,sep=',')
#%%
N = 21250
data_M = np.array([[]]*512)
for i in range(N):

    data = pd.read_table(r"D:\Study\MRes study material\2nd project\fdtd\training_data\results_1\results_{0}.txt".format(1), sep='\t',
                         header=None)
    data = np.array(data)[:,0].reshape(512,1)
    data_M = np.append(data_M, data, axis=1)
    print('\r', i, end="",
          flush=True)
data_M = data_M *1e9
dataframe = pd.DataFrame(data_M)
dataframe.to_csv("range.csv",index=False,header=False,sep=',')


#%%

x_tar= gaussian_distibution(center=226, sigma=100, length = 512)
x_tar=(x_tar+0.0003)*200
# x_tar[236]=1
# x_tar[221]=0.5
# x_tar[251]=0.5
plt.grid()
plt.plot( x_tar[0,:])
plt.show()

plt.figure(figsize=(3, 3))
plt.specgram(x_tar[0,:])
#
plt.axis('off')
plt.savefig('x_tar',bbox_inches='tight',pad_inches=0.0)
#%%
# x_tar = data
x_tar[:280,1]=data[:280,1]
x_tar[320:512,1]=data[320:512,1]
plt.plot(x_tar[:, 0] * 1e9, x_tar[:, 1])

plt.show()

#%%
I = Image.open('./x_tar.png')

I_array = np.array(I)
I_array = tf.image.rgb_to_grayscale(I_array[:,:,:3])

I_array = I_array/255


x_tar=x_tar.reshape(1,128,4,1)
data = np.append(I_array,x_tar, axis=2)