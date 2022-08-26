import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.keras import layers, models, regularizers
import ipykernel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
# from sklearn.decomposition import PCA
from numpy import random
from sklearn.preprocessing import MinMaxScaler

# tf.test.is_gpu_available()
# tf.config.experimental.list_physical_devices('GPU')

# def R2_i(y_true, y_pred):
#
#     SS_res =  K.sum(K.square( y_true-y_pred ),1)
#     avg = K.transpose([K.mean(y_true, 1)]*10 )
#     SS_tot = K.sum(K.square( y_true - avg),1 )
#     r2 = SS_res/SS_tot*15
#
#     return K.mean(r2)
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


def inv_R2_all(y_true, y_pred):

    SS_res =  K.sum(K.square( y_true-y_pred ),1)
    avg = K.transpose([K.mean(y_true, 1)]*10 )
    SS_tot = K.sum(K.square( y_true - avg),1 )
    loss = K.square(-0.1 + SS_res / (SS_tot + K.epsilon()))
    return K.sum(loss)

# def S(y_true, y_pred):
#
#     return K.mean( 1-abs(y_true-y_pred)/y_true)



#%%
# Spec = np.array(pd.read_csv('training_T.csv', header=None)).T
# Geo = np.array(pd.read_csv('Geo_T.csv', header=None))
#
# i_M=[]
# for i in range(len(Geo)):
#     if Geo[i,0]<600:
#         i_M.append(i)
#     print('\r',i, end="",
#           flush=True)
# i_M = np.array(i_M)
#
# Geo = Geo[i_M,:]
# Spec = Spec[i_M,:]
#%%
# x_M = np.array([[]]*512).reshape(0,512)
# y_M = np.array([[]]*10)
# y_M = y_M.reshape(0,10)
# i_M=[]
# for i in range(len(Geo)):
#     if max(spec[i]) >0.5 :
#         i_M.append(i)
#         x_M = np.append(x_M, spec[i:i+1,:], axis=0)
#         y_M = np.append(y_M, Geo[i:i+1,:], axis=0)
#     print('\r',i, end="",
#           flush=True)
# i_M = np.array(i_M)
#
# spec = x_M
# Geo = y_M



#%%
Geo = np.load(file="Geo_600.npy")
Spec = np.load(file="wave.npy")

#%%
idx = np.load(file="idx.npy")


Geo = Geo[idx,:]
spec = Spec[idx,:]

#%%
N = 22007
x_train = spec[:N:]
x_test = spec[N:,:]
y_train = Geo[:N,:]
y_test = Geo[N:,:]
#%%
# pca = PCA(n_components = 0.95)
# x_train = pca.fit_transform(x_train)
# # x_train = pca.transform(x_train)
# x_test = pca.transform(x_test)


#%%
# scaler = MinMaxScaler()
# scaler.fit(y_train)
# y_train = scaler.transform(y_train)
# y_test = scaler.transform(y_test)

#%%
reg = 1e-3
model = models.Sequential()

model.add(layers.Input((512)))

model.add(layers.Dense(1024,kernel_regularizer=regularizers.l2(reg)))
model.add(tf.keras.layers.BatchNormalization(trainable=True))
model.add(layers.Activation('relu'))

model.add(layers.Dense(2048,kernel_regularizer=regularizers.l2(reg)))
model.add(tf.keras.layers.BatchNormalization(trainable=True))
model.add(layers.Activation('relu'))

model.add(layers.Dense(1024,kernel_regularizer=regularizers.l2(reg)))
model.add(tf.keras.layers.BatchNormalization(trainable=True))
model.add(layers.Activation('relu'))

model.add(layers.Dense(512,kernel_regularizer=regularizers.l2(reg)))
model.add(tf.keras.layers.BatchNormalization(trainable=True))
model.add(layers.Activation('relu'))

model.add(layers.Dense(128,kernel_regularizer=regularizers.l2(reg)))
model.add(tf.keras.layers.BatchNormalization(trainable=True))
model.add(layers.Activation('relu'))

model.add(layers.Dense(64,kernel_regularizer=regularizers.l2(reg)))
model.add(tf.keras.layers.BatchNormalization(trainable=True))
model.add(layers.Activation('relu'))

model.add(layers.Dense(10,kernel_regularizer=regularizers.l2(reg)))


model.summary()
#%%
opt = tf.keras.optimizers.Adam(1e-3)

checkpoint_filepath = './tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_TR',
    mode='max',
    save_best_only=True)

model.compile(optimizer=opt,
              loss= tf.keras.losses.Huber(),#tf.keras.losses.Huber(),tf.keras.losses.LogCosh()
              metrics=[TR,R2])

#%%
history = model.fit(x_train, y_train, epochs=200,batch_size=512,
                    validation_data=(x_test, y_test)) #,callbacks=[model_checkpoint_callback]
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
plt.grid()
plt.show()

#%%
r = np.zeros(x_train.shape[0])
predictions = model.predict(x_train)
# r2_score(y_test.flatten(), predictions.flatten())

for k in range(x_train.shape[0]-10):
    r2 = r2_score(y_train.T[:,k], predictions.T[:,k])
    r[k]=r2

np.mean(r)

#%%
plt.plot(np.array(range(0,10)).reshape(10,1),y_test.T)
plt.plot(np.array(range(0,10)).reshape(10,1),predictions.T)
plt.show()

#%%
# inv_design = model.predict(x_train[3097,:].reshape(1,512))
# r2_score(y_train[3097,:].flatten(), inv_design.flatten())
inv_design = model.predict(x_test)
# inv_design = model.predict(x_tar[:, 1].reshape(1,512))
r2_score(y_test, inv_design)
#%%
data_M = np.array([[]]*512).T
label_M = np.array([[]]*10).T
r_M = np.array([])
for i in range(x_test.shape[0]):
    if r[i]<0.90:
        data_M = np.append(data_M, x_test[i:i+1,:], axis=0)
        r_M = np.append(r_M, r[i:i+1], axis=0)
        label_M = np.append(label_M, y_test [i:i+1,:], axis=0)

len(r_M)
#%%
x_train = np.append(x_train, data_M, axis=0)
y_train = np.append(y_train, label_M, axis=0)
#%%
plt.plot(y_train[-1,:])

plt.plot(inv_design.flatten())

plt.show()

#%%
print(max(history.history['val_TR']))
print((history.history['val_R2'][np.argmax(history.history['val_TR'])]))



