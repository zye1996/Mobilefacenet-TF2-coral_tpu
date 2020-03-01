import tensorflow as tf
import tensorflow.keras as keras
import math

# Bottleneck
class Bottleneck(keras.layers.Layer):

    def __init__(self, d_in, d_out, stride, depth_multiplier):
        super(Bottleneck, self).__init__()

        # decide whether there would be a short cut
        if d_in == d_out and stride == 1:
            self.connect = True
        else:
            self.connect = False

        # conv
        self.conv = keras.models.Sequential([
            # point-wise layers
            keras.layers.Conv2D(d_in * depth_multiplier, kernel_size=1, strides=1, padding='VALID', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.PReLU(),

            # depth-wise layers
            keras.layers.ZeroPadding2D(padding=(1, 1)), # manually padding
            keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='VALID', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.PReLU(),

            # point-wise layers linear
            keras.layers.Conv2D(d_out, kernel_size=1, strides=1, padding='VALID', use_bias=False),
            keras.layers.BatchNormalization()])

    def call(self, inputs, **kwargs):
        if self.connect:
            return self.conv(inputs) + inputs
        else:
            return self.conv(inputs)


# conv block
class ConvBlock(keras.layers.Layer):

    def __init__(self, d_in, d_out, kernel_size, stride, padding, depthwise=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear

        # padding layer
        if padding != 0:
            self.padding = keras.layers.ZeroPadding2D(padding=(padding, padding))
        else:
            self.padding = None

        # conv layer
        if depthwise:
            self.conv = keras.layers.DepthwiseConv2D(kernel_size, strides=stride, padding='VALID', use_bias=False)
        else:
            self.conv = keras.layers.Conv2D(d_out, kernel_size, strides=stride, padding='VALID', use_bias=False)

        # batch norm
        self.bn = keras.layers.BatchNormalization()

        if not self.linear:
            self.prelu = keras.layers.PReLU()


    def call(self, inputs, **kwargs):

        if self.padding is not None:
            x = self.padding(inputs)
        else:
            x = inputs

        x = self.conv(x)
        x = self.bn(x)

        if not self.linear:
            return self.prelu(x)
        else:
            return x


# arc face loss calculation
class ArcFace(keras.layers.Layer):

    def __init__(self, n_classes=10, s=32.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs, **kwargs):
        x, y = inputs
        c = tf.shape(x)[-1]
        # normalize feature
        x = tf.math.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.math.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(tf.clip_by_value(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]


class MobileFacenet(keras.models.Model):

    def __init__(self, setting=Mobilefacenet_bottleneck_setting):

        super(MobileFacenet, self).__init__()

        # layer 1
        self.conv1 = ConvBlock(d_in=3, d_out=64, kernel_size=3, stride=2, padding=1)
        # layer 2
        self.dwconv1 = ConvBlock(d_in=64, d_out=64, kernel_size=3, stride=1, padding=1, depthwise=True)
        # blocks
        self.inplanes = 64
        self.block = keras.models.Sequential()
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    self.block.add(Bottleneck(self.inplanes, c, s, t))
                else:
                    self.block.add(Bottleneck(self.inplanes, c, 1, t))
        # layer 3
        self.conv2 = ConvBlock(d_in=128, d_out=512, kernel_size=1, stride=1, padding=0) # no padding
        # layer 4
        self.linear_conv1 = ConvBlock(d_in=512, d_out=512, kernel_size=(7, 6), stride=1, padding=0,
                                      depthwise=True, linear=True)
        # layer 5
        self.linear_conv2 = ConvBlock(d_in=512, d_out=128, kernel_size=1, stride=1, padding=0,
                                      linear=True)

        # layer output
        self.out_layer = ArcFace()
    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.conv1(x)
        x = self.dwconv1(x)
        x = self.block(x)
        x = self.conv2(x)
        x = self.linear_conv1(x)
        x = self.linear_conv2(x)

        return x

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)






if __name__ == '__main__':
    model = MobileFacenet()
    model.build_graph(input_shape=(1, 112, 96, 3))
    print(model.summary())
