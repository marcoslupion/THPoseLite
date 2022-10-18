# Imports

import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Input, Conv2D, ReLU, BatchNormalization,Add, AveragePooling2D, Flatten, Dense,Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import LeakyReLU, Conv2DTranspose, Concatenate
from keras import Sequential
import argparse

# General parameters

landmarks = 22
coordinates = 3


# ResNetV50 functions

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(dropout,x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    if dropout: y = Dropout(0.5)(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def ResNet50(dropout=True):

    inputs = Input(shape=(120, 160, 3))
    num_filters = 32

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    num_blocks_list = [2,2,2,2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(dropout,t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2

    t = AveragePooling2D(2)(t)
    t = Flatten()(t)

    t = Dense(2048, activation='relu', kernel_regularizer='l2')(t)
    t = Dropout(0.2)(t)
    t = Dense(512, activation='relu', kernel_regularizer='l2')(t)
    t = Dropout(0.2)(t)
    t = Dense(128, activation='relu', kernel_regularizer='l2')(t)
    t = Dropout(0.2)(t)
    outputs = Dense( (landmarks * coordinates)  , dtype='float32')(t)

    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

    return model

# U-NET Functions

def downsample(filters, size, apply_batchnorm=True, apply_dropout=False, strides = 2, padding = "same"):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = Sequential()
  result.add(Conv2D(filters, size, strides= strides, padding=padding,kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(BatchNormalization())
  if apply_dropout:
    result.add(Dropout(0.1))
  result.add(LeakyReLU())

  return result
def upsample(filters, size, apply_dropout=False, strides = 2, padding = "same"):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = Sequential()
  result.add(Conv2DTranspose(filters, size, strides=strides,padding=padding,kernel_initializer=initializer,
                                    use_bias=False))

  result.add(BatchNormalization())

  if apply_dropout:
      result.add(Dropout(0.5))
  result.add(ReLU())
  return result

def UNET():
  inputs = Input(shape=[120, 160, 3])

  down_stack = [
    downsample(64, 2, apply_batchnorm=False),  # (batch_size, 60, 80, 64)
    downsample(128, 2,apply_dropout=False, strides = (3,4)),  # (batch_size, 20, 20, 128)
    downsample(256, 4,apply_dropout=False),  # (batch_size, 10, 10, 256)
    downsample(512, 3,apply_dropout=False, padding = "valid", strides = 1),  # (batch_size, 8, 8, 512)
    downsample(512, 4,apply_dropout=False),  # (batch_size, 4, 4, 512)
    downsample(512, 4,apply_dropout=False),  # (batch_size, 2, 2, 512)
    downsample(512, 4,apply_dropout=False),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 3, apply_dropout=True, padding = "valid", strides = 1),  # (batch_size, 10, 10, 512)
    upsample(256, 4),  # (batch_size, 20, 20, 256)
    upsample(64, 2,strides = (3,4)),  # (batch_size, 128, 128, 128)
    upsample(128, 2)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer, activation = "relu")  # (batch_size, 256, 256, 3)

  x = inputs

  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = Concatenate()([x, skip])
  x = last(x)

  x = Flatten()(x)

  x = Dense(2048, activation='relu', kernel_regularizer='l2')(x)
  x = Dropout(0.2)(x)
  x = Dense(512, activation='relu', kernel_regularizer='l2')(x)
  x = Dropout(0.2)(x)
  x = Dense(128, activation='relu', kernel_regularizer='l2')(x)
  x = Dropout(0.2)(x)
  outputs = Dense((landmarks * coordinates) , dtype='float32')(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
  return model

# MobileNetV2

def MobileNetV2():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(120,160,3),include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu', trainable=True)(x)
    x = Dropout(0.1)(x)
    x = Dense(512, activation='relu', trainable=True)(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu', trainable=True)(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu', trainable=True)(x)

    predictions = Dense( (landmarks * coordinates) , dtype='float32')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
      layer.trainable = True
    model.compile(
          optimizer='adam',
          loss='mean_squared_error',
          metrics=[tf.keras.metrics.RootMeanSquaredError()]
      )
    print(model.summary())
    return model



# Command line parameters

parser = argparse.ArgumentParser(description='Choose the type of model')
parser.add_argument('--name', dest='model_name', type=str, help='Model Name')

args = parser.parse_args()
try:
    model_name = args.model_name
except:
    print("Error. The parameter --name has to be set. The possible values are the following:\n")
    print("\t\tResNet50\n\t\tUNET\n\t\tMobileNetV2")
    exit(0)

if model_name not in ["ResNet50", "UNET","MobileNetV2"]:
    print("Error. The possible values are the following:\n")
    print("\t\tResNet50\n\t\tUNET\n\t\tMobileNetV2")
    exit(0)

# Creation of the models 

if model_name == "ResNet50":
    model = ResNet50()
elif model_name == "UNET":
    model = UNET()
else:
    model = MobileNetV2

print(model.summary())

#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', patience=30)
#my_callbacks = [early_stopping]
#model.fit(train_dataset, epochs=1000, validation_data=test_dataset , callbacks=my_callbacks)

