import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
# import wandb
from tensorflow import keras
import config
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

# wandb.init(project="ML-final-project", entity="andr3us")


batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  config.data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(config.img_height, config.img_width),
  batch_size=config.batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
config.data_dir,
validation_split=0.2,
subset="validation",
seed=123,
image_size=(config.img_height, config.img_width),
batch_size=config.batch_size)


# Normalizing the data

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


def resblock(x, filters, kernelsize):
    fx = tf.keras.layers.Conv2D(filters, kernelsize, activation='relu', padding='same')(x)
    fx = tf.keras.layers.BatchNormalization()(fx)
    fx = tf.keras.layers.Conv2D(filters, kernelsize, padding='same')(fx)
    out = tf.keras.layers.Add()([x,fx])
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.BatchNormalization()(out)
    return out

def convblock2d(x, filters, kernelsize: int, num_convs: int, activation="relu", padding="same"):
    if num_convs <= 0:
        return x
    out = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernelsize, activation="relu", padding=padding)(x)
    for i in range(num_convs - 1):
        out = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernelsize, activation="relu", padding=padding)(out)
    return out


inputs = tf.keras.Input(shape=(180, 180, 3))
hidden_layer = convblock2d(inputs, 128, 3, 1)
hidden_layer = resblock(hidden_layer, 128, 3)
hidden_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", strides=2)(hidden_layer)
hidden_layer = convblock2d(hidden_layer, 64, 3, 1)
hidden_layer = resblock(hidden_layer, 64, 3)
hidden_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", strides=2)(hidden_layer)
hidden_layer = convblock2d(hidden_layer, 32, 3, 1)
hidden_layer = resblock(hidden_layer, 32, 3)
hidden_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", strides=2)(hidden_layer)
hidden_layer = convblock2d(hidden_layer, 16, 3, 1)
hidden_layer = resblock(hidden_layer, 16, 3)
hidden_layer = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu", strides=2)(hidden_layer)
hidden_layer = convblock2d(hidden_layer, 8, 3, 1)
hidden_layer = resblock(hidden_layer, 8, 3)
hidden_layer = tf.keras.layers.Flatten()(hidden_layer)
output_layer = tf.keras.layers.Dense(10)(hidden_layer)

model = keras.Model(inputs=inputs, outputs=output_layer, name="my-model")
model.summary()


print(tf.config.list_physical_devices('GPU')) # True/False


model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
  loss=tf.keras.losses.BinaryCrossentropy(),
  metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.BinaryCrossentropy()])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=30,
  batch_size=config.batch_size
  # callbacks=[WandbMetricsLogger(log_freq="batch"),
  # WandbModelCheckpoint("models", save_best_only=True),]
)

