import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from ViT_tools import Patches, PatchEncoder, mlp
from ResBlock import ResnetBlock

def create_vit_classifier(input_shape,patch_size,projection_dim,transformer_layers,num_heads,transformer_units,mlp_head_units):
    chl = 16
    inputs = layers.Input(shape=input_shape)
    conv_num = 2
    conv = inputs

    for _ in range(conv_num):

        conv = ResnetBlock(3, [0, chl])(conv)
        conv = ResnetBlock(3, [0, chl])(conv)
        conv = ResnetBlock(3, [0, chl])(conv)

        conv = layers.AveragePooling2D(pool_size=2)(conv)

    # Create patches.
    patches = Patches(patch_size)(conv)

    # Encode patches.
    num_patches = (conv.shape[1]// patch_size) ** 2
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = tfa.layers.MultiHeadAttention(
            head_size=projection_dim,num_heads=num_heads, dropout=0
        )([x1,x1,x1])
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    # representation = layers.Dropout(0.3)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0)
    # Classify outputs.
    logits = layers.Dense(10)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model