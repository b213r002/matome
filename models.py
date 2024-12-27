import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    class Bias(keras.layers.Layer):
        def __init__(self, input_shape):
            super(Bias, self).__init__()
            self.W = tf.Variable(initial_value=tf.zeros(input_shape), trainable=True)

        def call(self, inputs):
            return inputs + self.W

    model = keras.Sequential()
    model.add(layers.Permute((2, 3, 1), input_shape=(2, 8, 8)))
    for _ in range(11):  # 畳み込み層を11回追加
        model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.Conv2D(1, kernel_size=1, use_bias=False))
    model.add(layers.Flatten())
    model.add(Bias((64,)))  # 修正点
    model.add(layers.Activation('softmax'))

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    print('モデルは正常')  # 修正点
    return model



