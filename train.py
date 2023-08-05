import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from processData import processDirectory
from keras.layers import *
import keras
import keras.losses as ls
import numpy as np
from functools import partial

"""from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()"""

tf.config.experimental.set_virtual_device_configuration(tf.config.list_physical_devices('GPU')[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)
])

data_aug=tf.keras.Sequential([
    RandomBrightness(0.1),
    RandomContrast(0.1),
    RandomRotation(.2),
    RandomZoom(.1,.1)
])

print(tf.config.list_physical_devices('GPU'))
dataProp=partial(processDirectory,"memes")
data_gen = tf.data.Dataset.from_generator(dataProp, output_signature=(
    tf.TensorSpec(shape=(704, 704, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(704, 704, 3), dtype=tf.float32)
))
data_gen=data_gen.repeat()
data_gen=data_gen.batch(8)
data_gen=data_gen.map(lambda x,y:(data_aug(x),y))

"""def convBlock(input,filters=16,kernel_size=2,padding="same",pool=False):
    x=Conv2D(filters=filters,kernel_size=kernel_size,padding=padding,activation="relu")(input)
    if pool:x=MaxPooling2D()(x)
    x=BatchNormalization()(x)
    return x
def convTransposeBlock(input,filters=16,kernel_size=2,padding="same",upscale=False):
    x=input
    if upscale:x=UpSampling2D()(x)
    x=Conv2DTranspose(filters=filters,kernel_size=kernel_size,padding=padding)(x)
    x=BatchNormalization()(x)
    return x
encoderInput=Input(shape=(704,704,3))
x=convBlock(encoderInput,pool=True)
for i in range(5):
    x=convBlock(x,kernel_size=i+1)
    x2=convBlock(x,kernel_size=i+1)
    x=x+x2
    x=LayerNormalization()(x)
    x=convBlock(x,filters=8,pool=True)
encoder=keras.Model(inputs=encoderInput,outputs=x)
inp=Input((11,11,8))
x=convTransposeBlock(inp,32,kernel_size=4)
for i in range(5,0,-1):
    x=convTransposeBlock(x,kernel_size=i,upscale=True)
    x2=convTransposeBlock(x,kernel_size=i)
    x=x+x2
    x=LayerNormalization()(x)
    x=convTransposeBlock(x,kernel_size=i)
x=convTransposeBlock(x,upscale=True)
x=convTransposeBlock(x)
x=Conv2D(3,2,activation="sigmoid",padding="same")(x)
decoder=keras.Model(inputs=inp,outputs=x)
decoder.build((None,11,11,8))
print(decoder.summary())
encoder.build((None,704,704,3))
encoder.summary()"""
class AutoEncoder(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder=keras.Sequential([
            Input((704,704,3)),
            Conv2D(16,1,padding="same",activation="relu"),
            Conv2D(16,2,padding="same",activation="relu"),
            Conv2D(16,3,padding="same",activation="relu"),
            Conv2D(16,4,padding="same",activation="relu"),
            MaxPooling2D((2,2)),
            Conv2D(16,5,padding="same",activation="relu"),
            MaxPooling2D((2,2)),
            Conv2D(32,6,padding="same",activation="relu"),
            MaxPooling2D((2,2)),
            Conv2D(32,7,padding="same",activation="relu"),
            MaxPooling2D((2,2)),
            Conv2D(32,8,padding="same",activation="relu"),
        ])
        self.decoder=keras.Sequential([
            Input((44,44,32)),
            Conv2DTranspose(32,9,padding="same",activation="relu"),
            UpSampling2D((2,2)),
            Conv2DTranspose(32,8,padding="same",activation="relu"),
            UpSampling2D((2,2)),
            Conv2DTranspose(32,7,padding="same",activation="relu"),
            UpSampling2D((2,2)),
            Conv2DTranspose(32,6,padding="same",activation="relu"),
            UpSampling2D((2,2)),
            Conv2DTranspose(16,5,padding="same",activation="relu"),
            Conv2DTranspose(16,4,padding="same",activation="relu"),
            Conv2DTranspose(16,3,padding="same",activation="relu"),
            Conv2DTranspose(16,2,padding="same",activation="relu"),
            Conv2DTranspose(3,1,padding="same",activation="sigmoid"),
        ])
        """self.encoder=encoder
        self.decoder=decoder"""
        self.encoder.build((None,704,704,3))
        self.decoder.build((None,44,44,32))
        self.encoder.summary()
        self.decoder.summary()
    def call(self, inputs, training=None, mask=None):
        x=self.encoder(inputs)
        x=self.decoder(x)
        return x
ae=AutoEncoder()
ae.build(((None,704,704,3)))
ae.compile(
    tf.keras.optimizers.Adam(),
    loss=ls.MeanSquaredError(),
    metrics=["MAE"]
)
ae.summary()
ae.fit(data_gen,steps_per_epoch=394, epochs=20,use_multiprocessing=True)
ae.save("saved/model5")
print("Model saved")