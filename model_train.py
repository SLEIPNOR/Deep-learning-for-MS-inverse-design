import ipykernel
import numpy as np
import pandas as pd
import basic_experts_model as bem
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import NN_model as NNM
import tensorflow_addons as tfa
import math
from tensorflow.keras.layers import GaussianNoise
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

def det(y_true,y_pred):
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    return r2
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def down_sampling(images, size):

    images_d = []
    for i in range(len(images)):
        images_d.append(np.array(tf.image.resize(images[i], [size[0], size[1]])))
        print('\r'"down sampling:{0}%".format(round((i + 1) * 100 / len(images)))
              , end="", flush=True)
    images_d = np.array(images_d)
    return images_d

def inv_R2_all(y_true, y_pred):

    SS_res =  K.sum(K.square( y_true-y_pred ),1)
    avg = K.transpose([K.mean(y_true, 1)]*10 )
    SS_tot = K.sum(K.square( y_true - avg),1 )
    loss = K.square( -0.09+SS_res / (SS_tot + K.epsilon()))
    return K.sum(loss)

def TR(y_true, y_pred):

    SS_res =  K.sum(K.square( y_true-y_pred ),1)
    avg = K.transpose([K.mean(y_true, 1)]*10 )
    SS_tot = K.sum(K.square( y_true - avg),1 )
    pass_rate = K.mean(K.round(1 -0.2- SS_res/(SS_tot+K.epsilon())))
    return pass_rate

def R2(y_true, y_pred):

    SS_res =  K.sum(K.square( y_true-y_pred ),1)
    avg = K.transpose([K.mean(y_true, 1)]*10 )
    SS_tot = K.sum(K.square( y_true - avg),1 )
    return K.mean( 1 - SS_res/(SS_tot+K.epsilon()) )

def S(y_true, y_pred):

    return  1-abs(K.sum(y_true)-K.sum(y_pred))/K.sum(y_true)


def model_training(train_images, test_images, train_labels, test_labels,ep):

    # building model
    opt = tf.keras.optimizers.Adam(1e-3)

    #tf.keras.losses.log_cosh
    model.compile(optimizer = opt,
                  loss = [inv_R2_all] , # tf.keras.losses.Huber()
                  metrics=[TR,R2],
                  )
    checkpoint_filepath = './tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_TR',
        mode='max',
        save_best_only=True)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                   patience=500, verbose=0, mode='min',
                                                   baseline=None, restore_best_weights=True)




    history = model.fit(train_images, train_labels,epochs=ep,batch_size=512,
                        validation_data=(test_images, test_labels),callbacks=[model_checkpoint_callback])

#sample_weight=weight_tf

    return model, history

#%% Mix data generate

# img_Mat = []
# N = 36250
#
# for i in range(N):
#
#     I = Image.open('TF/TF_{0}.png'.format(i))
#
#     I_array = np.array(I)
#     I_array = tf.image.rgb_to_grayscale(I_array[:,:,:3])
#     I_array = np.array(tf.image.resize(I_array, [128, 128]))
#     img_Mat.append(I_array/255)
#     print('\r'"loading images:{0}%".format(round((i + 1) * 100 / N)), end="",
#           flush=True)
#
# # img_Mat = down_sampling(img_Mat, 128)
# img_Mat = np.array(img_Mat)
# np.save(file="image.npy", arr=img_Mat)
#
# img_Mat = np.load(file="image.npy")
# img_Mat = down_sampling(img_Mat, [128,128])
# Spec= np.array(pd.read_csv('training_T.csv', header=None))
# Spec=Spec.T
#%%
#
# Spec=Spec.reshape(36250,128,4,1)
# data = np.append(img_Mat,Spec, axis=2)
# np.save(file="MixData.npy", arr=data)

#%% spec+sampling point

# Spec= np.array(pd.read_csv('testing_sp.csv', header=None))
# # range= np.array(pd.read_csv('range.csv', header=None))/1000
#
# # Spec = Spec+range
# N = 1500
# Spec=Spec.T
# Spec=Spec.reshape(N,32,16,1)
# Spec = np.append(Spec,Spec, axis=2)
# np.save(file="SpecData_testing.npy", arr=Spec)

#%% Data reading
Geo = np.load(file="Geo_600.npy")
Spec = np.load(file="SpecData_600.npy")
#%%
# Spec = down_sampling(Spec,[32,32])
# np.save(file="MixData_600_64.npy", arr=Spec )
#%%
# Spec= np.array(pd.read_csv('training.csv', header=None)).T
# Spec = down_sampling(Spec, [32,32])
# Spec = np.load(file="image.npy").reshape(16250,128,128,3)
# spec = np.array(pd.read_csv('training.csv', header=None)).T

#%%
# idx = tf.range(len(Geo))
#
# idx = np.array(tf.random.shuffle(idx))
# np.save(file="idx.npy", arr=idx)

#%%

idx = np.load(file="idx.npy")

# Geo = Geo[idx,:]
# Spec = Spec[idx,:]

Geo = Geo[idx,:]
Spec = Spec[idx,:,:,:]

#%%

# i_M=[]
# for i in range(len(Geo)):
#     if Geo[i,0]< 600:
#         i_M.append(i)
#     print('\r',i, end="",
#           flush=True)
# i_M = np.array(i_M)

#%%
# Geo = Geo[i_M ,:]
# # Geo = sigmoid(Geo/600)
# Spec = Spec[i_M ,:,:,:]
#%%
# #
# Spec = down_sampling(Spec,64)
#%%
N =22007
# x_train = Spec[:N,:]
# x_test = Spec[N:,:]

x_train = Spec[:N,:,:,:]
x_test = Spec[N:,:,:,:]

y_train = Geo[:N,:]

y_test = Geo[N:,:]

# del Geo, Spec
#%%
# scaler = MinMaxScaler()
# scaler.fit(x_train.reshape(N,32*32))
# x_train = scaler.transform(x_train.reshape(N,32*32))
# x_test = scaler.transform(x_test.reshape(27509-N,32*32))
# x_train = x_train.reshape(N,32,32,1)
# x_test = x_test.reshape(27509-N,32,32,1)
#%%

model = bem.ResNet_expert(x_train)
# model = NNM.NN(x_train)
model.summary()

#%%
# model.load_weights('./checkpoints/my_checkpoint{}'.format(1))
model.load_weights('./CNN_2/tmp/checkpoint')

#%%

model, history = model_training(x_train, x_test,
                                    y_train, y_test,50)

#%% feedback training


# weight_tf_raw = np.array([1.0] * N)
# weight_tf=weight_tf_raw/np.sum(weight_tf_raw)
# # i_count =[]
# ep = 300
# T = 1
# for t in range(T):
#
#     # lr = 0.01
#
#     model, history = model_training(x_train, x_test,
#                                     y_train, y_test,weight_tf,ep)
#
#     r_t = np.empty(x_test.shape[0])
#     predictions_t = model.predict(x_test)
#     # r2_score(y_test.flatten(), predictions.flatten())
#
#     for k in range(x_test.shape[0]):
#         r2_t = r2_score(y_test.T[:, k], predictions_t.T[:, k])
#         r_t[k] = r2_t
#
#     print(r_t)
#
#     bad_data =[]
#     for l in range(len(r_t)):
#
#         if r_t[l] < 0.70:
#             bad_data.append(l)
#
#
#     r = np.empty(Spec[:N,:,:,:].shape[0])
#     predictions = model.predict(Spec[:N,:,:,:])
#
#     for k in range(Spec[:N,:,:,:].shape[0]):
#         r2 = r2_score(Geo[:N,:].T[:, k], predictions.T[:, k])
#         r[k] = r2
#
#
#     r_M = np.array([])
#
#     for i in range(Spec[:N,:,:,:].shape[0]):
#         # num = 0
#         if r[i] < 0.80:
#             # i_count.append(i)
#             # for ic_index in i_count:
#             #     if ic_index==i:
#             #         num = num+1
#
#             r_M = np.append(r_M, r[i:i + 1], axis=0)
#
#         weight_tf_raw[i] = min(max(weight_tf_raw[i] - (r[i]-0.85)*2,2e-1),5)
#     weight_tf = weight_tf_raw / np.sum(weight_tf_raw)
#     ep = 10
#
#     if len(r_M)==0:
#         break
#
#     # x_train = fd_M
#     # y_train = label_M
#     # sample = GaussianNoise(0.2)
#     # fd_M = np.array(sample(fd_M.astype(np.float32), training=True))




#%%

# plt.imshow(Spec[1340,:,:,:],cmap='gray')
# plt.show()


#%% training data validate
# r = np.empty(x_train.shape[0])
# predictions = model.predict(x_train)
# # r2_score(y_test.flatten(), predictions.flatten())
#
# for k in range(x_train.shape[0]):
#     r2 = r2_score(y_train.T[:,k], predictions.T[:,k])
#     r[k]=r2
#
# np.mean(r)

#%%

# fd_M = []
# r_M = np.array([])
# label_M = np.array([[]]*10).T
#
# for i in range(x_train.shape[0]):
#     if r[i]<0.8:
#         r_M = np.append(r_M, r[i:i+1], axis=0)
#         label_M = np.append(label_M, y_train[i:i + 1, :], axis=0)
#         fd_M.append(x_train[i,:,:,:])
#
# fd_M = np.array(fd_M)
#%%
# for i in range(1):
#     x_train = np.append(x_train, fd_M, axis=0)
#     y_train = np.append(y_train, label_M, axis=0)
# len(r_M)

#%% testing data validate
x_test_ex = np.load(file='SpecData_testing.npy')
y_test_ex = np.load(file='Geo_test.npy')
r_t = np.empty(x_test_ex.shape[0])
predictions_t = model.predict(x_test_ex)
# r2_score(y_test.flatten(), predictions.flatten())
a = 0
for k in range(x_test_ex.shape[0]):
    r2_t = r2_score(y_test_ex.T[:,k], predictions_t.T[:,k])
    if r2_t > 0.70:
        a = a+1
    r_t[k]=r2_t

rate = a/x_test_ex.shape[0]
rate


#%%
# model.save_weights('./checkpoints/my_checkpoint{}'.format(1))

#%%
# inv_design = model.predict(x_train[5000:,:,:,:])

#%%
# plt.plot(y_train[-1:])
# plt.plot(inv_design.flatten())
# plt.show()

#%%
# plt.imshow(Spec[10,:,:,:],cmap='gray')
# plt.show()
#%%
# data = pd.read_table(r"D:\Study\MRes study material\2nd project\fdtd\training_data\results_4\results_{0}.txt".format(1), sep='\t',
#                          header=None)
# data = np.array(data)[:,0]
# plt.plot(data*1e9,Spec[10,:,:16,:].flatten())
#
# plt.show()
#%%
k = k
plt.plot(range(1,11),predictions_t.T[:,k],label='predicted geo paras R2 = {0}'.format(round(r2_score(y_test_ex.T[:,k], predictions_t.T[:,k]),4)))
plt.plot(range(1,11),y_test_ex.T[:,k],label='ture geo paras')
plt.legend(loc='upper right')
plt.xlim([1, 10])
plt.grid()
plt.show()

#%%

plt.plot(history.history['R2'], label='R2')
plt.plot(history.history['val_R2'], label = 'val_R2')
plt.plot(history.history['TR'], label='TR')
plt.plot(history.history['val_TR'], label = 'val_TR')
plt.xlabel('Epoch')
plt.ylabel('Rate')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')
plt.grid()
plt.show()



#%%
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
# plt.ylim([0.5, 1])
plt.legend(loc='upper right')
plt.show()
plt.grid()

#%%
# a = np.array([ 378.844+127,473.558+10,270.887-18,299.046+28,70.1366-20,162.17+37,-108.759-27,23.6614+50,74.0773+70,-20.6858+10])
# b = np.array([ 378.844+127,473.558+185,270.887-108,299.046-48,70.1366+20,162.17+97,-108.759-97,23.6614+50,74.0773-30,-20.6858+80])
# c = np.array([ 378.844+267,473.558+420,270.887,299.046+128,70.1366+120,162.17-97,-108.759+67,23.6614+50,74.0773+100,-20.6858+20])
# d = Geo.T[:,10]-168
# plt.plot(range(1,11),Geo.T[:,10],label='ture geo paras')
# # plt.plot(range(1,11),a, linestyle=":",label='predicted geo paras R2=0.91')
# # plt.plot(range(1,11),b,linestyle="-.",label='predicted geo paras R2=0.71')
# # plt.plot(range(1,11),c,linestyle="--",label='predicted geo paras R2=0.04')
# plt.plot(range(1,11),d,linestyle="--",label='predicted geo paras R2=-3.7')
# plt.xlim([1, 10])
# plt.grid()
# plt.legend(loc='upper right')
# plt.title('R2={0}'.format(round(r2_score(Geo[10,:], d.T),4)))
# plt.show()

#%%
# print(max(history.history['val_TR']))
# print((history.history['val_R2'][np.argmax(history.history['val_TR'])]))


#%%
k = 86
plt.plot(x_test_ex[k,:,:16,:].flatten())
plt.show()





