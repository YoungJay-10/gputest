# import tensorflow as tf
#
# print(tf.config.list_physical_devices('GPU'))
# print(tf.test.is_gpu_available())

import tensorflow as tf
import os
import numpy as np

# 加载数据集
# from keras.datasets import mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 这里加载数据集要去访问一个网址进行下载，由于众所周知的原因，这个网址是访问不了的
# 所以这里使用下载下来的本地数据集。
# 链接：https://pan.baidu.com/s/1rzqWrlsO7Zg2gtmCxYQZjA
# 提取码：ff9z
path='D:\WorkPart\Mywork\【研究生】\杂_input\数据集/mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)
