import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from ResBlock import ResnetBlock

def ResNet_expert(train_images):
    trainable = False
    reg = 1e-3
    S = train_images.shape
    model = models.Sequential()
    chl = 32
    # First 3*3*64 stride = 2 cov layer with pooling
    model.add(layers.Conv2D(chl, (3, 3), strides=1, input_shape=S[1:],kernel_regularizer=regularizers.l2(reg)))
    model.add(layers.BatchNormalization(trainable = trainable))
    model.add(layers.Activation(tf.nn.relu))#leaky_relu

    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # ResNet block 64

    model.add(ResnetBlock(3, [0, chl])) # [x, chl] x is boolean 0 or 1 whether change to same channel
    model.add(ResnetBlock(3, [0, chl]))
    model.add(ResnetBlock(3, [0, chl]))

    model.add(layers.AveragePooling2D(pool_size=(2, 2)))



    model.add(ResnetBlock(3, [0, chl]))
    model.add(ResnetBlock(3, [0, chl]))
    model.add(ResnetBlock(3, [0, chl]))

    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # model.add(ResnetBlock(3, [0, chl]))
    # model.add(ResnetBlock(3, [0, chl]))
    # model.add(ResnetBlock(3, [0, chl]))
    #
    # model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    # model.add(layers.Activation('relu'))
    # model.add(layers.Dropout(0.3))

    # model.add(layers.Dense(32,kernel_regularizer=regularizers.l2(reg)))
    # model.add(tf.keras.layers.BatchNormalization(trainable = trainable))
    # model.add(layers.Activation('relu'))

    model.add(layers.Dense(10))
    # model.summary()
    # model.add(layers.Activation('sigmoid'))

    return model

