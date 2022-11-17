import tensorflow as tf
from tensorflow.keras import layers, Model


class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.drop = layers.Dropout(0.2)
    @tf.function
    def call(self, input1, input2):
        x = self.drop(input1)
        y = self.drop(input2)
        return tf.concat([x, y], axis=1)

training = True


model = MyModel()
x = tf.constant([[0.1, 0.1, 0.1, 0.1]])
y = tf.constant([[0.2, 0.2, 0.2, 0.2]])
z = model(x, y, training = True)
print(z)