import numpy as np
import pandas as pd
import basic_experts_model as bem
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from ViT_tools import Patches
import tensorflow_addons as tfa
from TFRes import create_vit_classifier
from tensorflow.keras import layers
import ipykernel
from ViT_tools import Patches, PatchEncoder, mlp
from sklearn.preprocessing import MinMaxScaler

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

def run_experiment(train_images, test_images, train_labels, test_labels,ep):

    optimizer = tfa.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=0
    )

    model.compile(
        optimizer=optimizer,
        loss= [inv_R2_all] ,#tf.keras.losses.Huber()
        metrics=[TR,R2],
    )

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                   patience=100, verbose=0, mode='min',
                                                   baseline=None, restore_best_weights=True)

    checkpoint_filepath = './tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_TR',
        mode='max',
        save_best_only=True)

    history = model.fit(
        train_images,
        train_labels,
        batch_size=512,
        epochs=ep,
        validation_data=(test_images, test_labels),
        callbacks=[model_checkpoint_callback]
        ) #


    return model, history

#%%
Geo = np.load(file="Geo_600.npy")
# Geo = np.array(pd.read_csv('Geo_T.csv', header=None))
Spec = np.load(file="SpecData_600.npy")
# Spec = down_sampling(Spec,[32,32])
#%%

# i_M=[]
# for i in range(len(Geo)):
#     if Geo[i,0]<600:
#         i_M.append(i)
#     print('\r',i, end="",
#           flush=True)
# i_M = np.array(i_M)
#
# Geo = Geo[i_M,:]
# Spec = Spec[i_M,:,:,:]
#
# np.save(file="Geo_600.npy", arr=Geo)
# np.save(file="MixData_600.npy", arr=Spec)
#%%
# idx = tf.range(len(Geo))
#
# idx = np.array(tf.random.shuffle(idx))
# np.save(file="idx.npy", arr=idx)

idx = np.load(file="idx.npy")
Spec = Spec[idx,:,:,:]
Geo = Geo[idx,:]
#%%

N= 22007


x_train = Spec[:N,:,:,:]
x_test = Spec[N:,:,:,:]

y_train = Geo[:N,:]

y_test = Geo[N:,:]


# scaler = MinMaxScaler()
# scaler.fit(x_train.reshape(N,32*32))
# x_train = scaler.transform(x_train.reshape(N,32*32))
# x_test = scaler.transform(x_test.reshape(27509-N,32*32))
# x_train = x_train.reshape(N,32,32,1)
# x_test = x_test.reshape(27509-N,32,32,1)


#%%

input_shape = x_train.shape[1:]
patch_size = 2
projection_dim = 16
transformer_layers = 8
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim
]
mlp_head_units = [64]





model = create_vit_classifier(input_shape,patch_size,projection_dim,transformer_layers,num_heads,transformer_units,mlp_head_units)

model.summary()
#%%
model.load_weights('./tmp/checkpoint')
#%%

model, history = run_experiment(x_train, x_test, y_train, y_test,50)
#%%
model.save_weights('./checkpoints/my_checkpoint{}'.format(1))
#%% testing data validate
x_test_ex = np.load(file='SpecData_testing.npy')
y_test_ex = np.load(file='Geo_test.npy')
r_t = np.empty(x_test_ex.shape[0])
predictions_t = model.predict(x_test_ex)
# r2_score(y_test.flatten(), predictions.flatten())
a = 0
for k in range(x_test_ex.shape[0]):
    r2_t = r2_score(y_test_ex.T[:,k], predictions_t.T[:,k])
    if r2_t > 0.50:
        a = a+1
    r_t[k]=r2_t

rate = a/x_test_ex.shape[0]
rate



#%%
plt.plot(x_train[0,:,:16,:].flatten())
plt.plot(x_test[8,:,:16,:].flatten())
plt.show()
#%%

# image_size = 72
# patch_size = 36
# #%%
# plt.figure(figsize=(4, 4))
# x_train = Spec[:N,:,:128,:]
# x_test = Spec[N:,:,:128,:]
# #%%
# image = x_train[np.random.choice(range(x_train.shape[0]))]
# plt.imshow(image.astype("float32"),cmap='gray')
# # plt.axis("off")
# plt.show()
# resized_image = tf.image.resize(
#     tf.convert_to_tensor([image]), size=(image_size, image_size)
# )
# plt.imshow(np.array(resized_image)[0,:,:,:],cmap='gray')
# # plt.axis("off")
# plt.show()
# #%%
# patches = Patches(patch_size)(resized_image)
#
# print(f"Image size: {image_size} X {image_size}")
# print(f"Patch size: {patch_size} X {patch_size}")
# print(f"Patches per image: {patches.shape[1]}")
# print(f"Elements per patch: {patches.shape[-1]}")
# #%%
# n = int(np.sqrt(patches.shape[1]))
# plt.figure(figsize=(4, 4))
# plt.subplots()
# for i, patch in enumerate(patches[0]):
#     ax = plt.subplot(n, n, i + 1)
#     patch_img = tf.reshape(patch, (patch_size, patch_size, 1))
#     plt.imshow(patch_img.numpy(),cmap='gray')
#     plt.axis("off")
# plt.show()



#%%
print(max(history.history['val_TR']))
print((history.history['val_R2'][np.argmax(history.history['val_TR'])]))
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
plt.ylim([0, 200])
plt.legend(loc='upper right')
plt.show()
plt.grid()

