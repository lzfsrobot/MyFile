from load_data import load_dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from    tensorflow.keras import layers, optimizers, datasets, Sequential
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# 加载数据
datas, labels = load_dataset(r'C:\Users\ASUS\Desktop\test')

# print(datas.shape, labels.shape)
# print(labels[0], labels[1000])

def preprocess(x, y):
    # [0~1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=2)
    return x,y

X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size = 0.2, random_state = random.randint(0, 100))
print(X_train.shape, y_train.shape)
print(y_test)
train_db = tf.data.Dataset.from_tensor_slices((X_train,y_train))
train_db = train_db.shuffle(500).map(preprocess).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((X_test,y_test))
test_db = test_db.map(preprocess).batch(64)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))
cnn_network = Sequential([
    # unit 1 64 => 32
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 2 32 => 16
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 3 16 => 8
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 4 8 => 4
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 5 4 = >2
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 6 2 => 1
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Flatten(), # 平坦层

    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(2, activation=None),
])

cnn_network.build(input_shape=[None, 64, 64, 3])
out = cnn_network(sample[0])

cnn_network.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy']) 
cnn_network.fit(train_db, validation_data=test_db, epochs=5) 

cnn_network.save(r'cnn_network.h5')

del cnn_network  
# 预测代码
model = tf.keras.models.load_model(r'cnn_network.h5')

sample = next(iter(train_db))
xx = sample[0]
yy = sample[1]
yy = tf.argmax(yy, axis=1)
pred = model.predict(xx)
pred = tf.argmax(pred, axis=1)

print(pred == yy)

