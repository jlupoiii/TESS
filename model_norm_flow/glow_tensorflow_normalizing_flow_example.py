# Code from https://github.com/simonwestberg/Glow



# TensorFlow
import tensorflow as tf
import tensorflow_probability as tf_prob
from tensorflow import keras
from tensorflow.keras.layers import Layer
import numpy as np

print('imported first batch')

### LAYERS

## ActNorm
class ActNorm(Layer):

    def __init__(self):
        super(ActNorm, self).__init__()

    # Create the state of the layer. ActNorm initialization is done in call()
    def build(self, input_shape):
        b, h, w, c = input_shape[0]  # Batch size, height, width, channels

        # Scale parameters per channel, called 's' in Glow
        self.scale = self.add_weight(
            shape=(1, 1, 1, c),
            trainable=True,
            name="actnorm_scale_" + str(np.random.randint(0, 1e6)))

        # Bias parameter per channel, called 'b' in Glow
        self.bias = self.add_weight(
            shape=(1, 1, 1, c),
            trainable=True,
            name="actnorm_bias_" + str(np.random.randint(0, 1e6))
        )

        # Used to check if scale and bias have been initialized
        self.initialized = self.add_weight(
            trainable=False,
            dtype=tf.bool,
            name="actnorm_initialized_" + str(np.random.randint(0, 1e6))
        )

        self.initialized.assign(False)

    @tf.function
    def call(self, inputs, forward=True):
        """
        inputs: list containing [input tensor, log_det]
        returns: output tensor
        """

        x = inputs[0]
        log_det = inputs[1]

        b, h, w, c = x.shape  # Batch size, height, width, channels

        if not self.initialized:
            """
            Given an initial batch X, setting
            scale = 1 / mean(X) and
            bias = -mean(X) / std(X)
            where mean and std is calculated per channel in X,
            results in post-actnorm activations X = X*s + b having zero mean
            and unit variance per channel.
            """

            assert (len(x.shape) == 4)

            # Calculate mean per channel
            mean = tf.math.reduce_mean(x, axis=[0, 1, 2], keepdims=True)

            # Calculate standard deviation per channel
            std = tf.math.reduce_std(x, axis=[0, 1, 2], keepdims=True)

            # Add small value to std to avoid division by zero
            eps = tf.constant(1e-6, shape=std.shape, dtype=std.dtype)
            std = tf.math.add(std, eps)

            self.scale.assign(tf.math.divide(1.0, std))
            self.bias.assign(-mean)

            self.initialized.assign(True)

        if forward:

            outputs = tf.math.multiply(x + self.bias, self.scale)

            # log-determinant of ActNorm layer
            log_s = tf.math.log(tf.math.abs(self.scale))
            log_det += h * w * tf.math.reduce_sum(log_s)

            return outputs, log_det

        else:
            # Reverse operation
            outputs = (x / self.scale) - self.bias

            # log-determinant of ActNorm layer
            log_s = tf.math.log(tf.math.abs(self.scale))
            log_det -= h * w * tf.math.reduce_sum(log_s)
            return outputs, log_det


## Permutation (1x1 convolution, reverse, or shuffle)
class Permutation(Layer):

    def __init__(self, perm_type="1x1"):
        super(Permutation, self).__init__()

        self.types = ["1x1", "reverse", "shuffle"]
        self.perm_type = perm_type

        if perm_type not in self.types:
            raise ValueError("Incorrect permutation type, should be either "
                             "'1x1', 'reverse', or 'shuffle'")

    # Create the state of the layer (weights)
    def build(self, input_shape):
        b, h, w, c = input_shape[0]

        if self.perm_type == "1x1":
            self.W = self.add_weight(
                shape=(1, 1, c, c),
                trainable=True,
                initializer=tf.keras.initializers.orthogonal,
                name="1x1_W_" + str(np.random.randint(0, 1e6))
            )
        elif self.perm_type == "reverse":
            pass

        elif self.perm_type == "shuffle":
            rng_seed = abs(hash(self.perm_type)) % 10000000
            self.indices = np.random.RandomState(seed=rng_seed).permutation(np.arange(c))
            self.reverse_indicies = [0] * c
            for i in range(c):
                self.reverse_indicies[self.indices[i]] = i

    # Defines the computation from inputs to outputs
    def call(self, inputs, forward=True):
        """
        inputs: input tensor or list containing [input tensor, log_det]
        returns: output tensor
        """

        x = inputs[0]
        log_det = inputs[1]

        b, h, w, c = x.shape

        if forward:
            if self.perm_type == "1x1":
                outputs = tf.nn.conv2d(x,
                                       self.W,
                                       strides=[1, 1, 1, 1],
                                       padding="SAME")

                # Log-determinant
                det = tf.math.reduce_sum(tf.linalg.det(self.W))
                log_det += h * w * tf.math.log(tf.math.abs(det))

                return outputs, log_det

            elif self.perm_type == "reverse":
                output = x[:, :, :, ::-1]
                log_det += 0

                return output, log_det

            elif self.perm_type == "shuffle":
                permuted_output = tf.gather(x, self.indices, axis=3)
                log_det += 0

                return permuted_output, log_det

        else:
            if self.perm_type == "1x1":
                W_inv = tf.linalg.inv(self.W)
                outputs = tf.nn.conv2d(x,
                                       W_inv,
                                       strides=[1, 1, 1, 1],
                                       padding="SAME")

                # Log-determinant
                det = tf.math.reduce_sum(tf.linalg.det(self.W))
                log_det -= h * w * tf.math.log(tf.math.abs(det))

                return outputs, log_det

            elif self.perm_type == "reverse":
                outputs = x[:, :, :, ::-1]

                log_det -= 0

                return outputs, log_det

            elif self.perm_type == "shuffle":
                reverse_permute_output = tf.gather(x, self.reverse_indicies, axis=3)

                log_det -= 0

                return reverse_permute_output, log_det


## Affine coupling
class AffineCoupling(Layer):

    def __init__(self, hidden_channels):
        """
        :param hidden_channels: Number of filters used for the hidden layers of the NN() function, see GLOW paper
        """
        super(AffineCoupling, self).__init__()

        self.NN = tf.keras.Sequential()
        self.hidden_channels = hidden_channels

    def build(self, input_shape):
        b, h, w, c = input_shape[0]

        self.NN.add(tf.keras.layers.Conv2D(self.hidden_channels, kernel_size=3,
                                           activation='relu', strides=(1, 1),
                                           padding='same'))
        self.NN.add(tf.keras.layers.Conv2D(self.hidden_channels, kernel_size=1,
                                           activation='relu', strides=(1, 1),
                                           padding='same'))
        self.NN.add(Conv2D_zeros(c))

    # Defines the computation from inputs to outputs
    def call(self, inputs, forward=True):
        """
        Computes the forward/reverse calculations of the affine coupling layer
        inputs: list containing [input tensor, log_det]
        returns: A tensor, same dimensions as input tensor, for next step of flow and the scalar log determinant
        """

        x = inputs[0]
        log_det = inputs[1]

        if forward:
            # split along the channels, which is axis=3
            x_a, x_b = tf.split(x, num_or_size_splits=2, axis=3)

            nn_output = self.NN(x_b)
            log_s = nn_output[:, :, :, 0::2]
            t = nn_output[:, :, :, 1::2]
            s = tf.math.sigmoid(log_s + 2)

            y_a = tf.math.multiply(s, x_a + t)

            y_b = x_b
            output = tf.concat((y_a, y_b), axis=3)

            _log_det = tf.math.log(tf.math.abs(s))
            _log_det = tf.math.reduce_sum(_log_det, axis=[1, 2, 3])
            log_det += tf.math.reduce_mean(_log_det)

            return output, log_det

        # the reverse calculations, if forward is False
        else:
            y_a, y_b = tf.split(x, num_or_size_splits=2, axis=3)

            nn_output = self.NN(y_b)
            log_s = nn_output[:, :, :, 0::2]
            t = nn_output[:, :, :, 1::2]

            s = tf.math.sigmoid(log_s + 2)

            x_a = tf.math.divide(y_a, s) - t
            x_b = y_b
            output = tf.concat((x_a, x_b), axis=3)

            _log_det = tf.math.log(tf.math.abs(s))
            _log_det = tf.math.reduce_sum(_log_det, axis=[1, 2, 3])
            log_det -= tf.math.reduce_mean(_log_det)

            return output, log_det


## Squeeze
class Squeeze(Layer):

    def __init__(self):
        super(Squeeze, self).__init__()

    # Defines the computation from inputs to outputs
    def call(self, inputs, forward=True):
        """
        inputs: input tensor
        returns: output tensor
        """
        if forward:
            outputs = tf.nn.space_to_depth(inputs, block_size=2)
        else:
            outputs = tf.nn.depth_to_space(inputs, block_size=2)

        return outputs


class Split(Layer):

    def __init__(self, split=True):
        super(Split, self).__init__()
        self.split = split

    def build(self, input_shape):
        b, h, w, c = input_shape

        # Uses learnt prior instead of fixed mean and std
        if self.split:
            self.prior = Conv2D_zeros(filters=c)
        else:
            self.prior = Conv2D_zeros(filters=2 * c)

    def call(self, inputs, forward=True, reconstruct=False):

        if forward:
            b, h, w, c = inputs.shape

            if self.split:
                x, z = tf.split(inputs, num_or_size_splits=2, axis=3)
                mean, log_sd = tf.split(self.prior(x), num_or_size_splits=2, axis=3)
                log_p = gaussian_logprob(z, mean, log_sd)
                log_p = tf.math.reduce_sum(log_p, axis=[1, 2, 3])
                log_p = tf.math.reduce_mean(log_p, axis=0)

                return x, z, log_p
            else:
                zero = tf.zeros_like(inputs)
                mean, log_sd = tf.split(self.prior(zero), num_or_size_splits=2, axis=3)
                log_p = gaussian_logprob(inputs, mean, log_sd)
                log_p = tf.math.reduce_sum(log_p, axis=[1, 2, 3])
                log_p = tf.math.reduce_mean(log_p, axis=0)
                z = inputs

                return z, log_p

        else:
            if reconstruct:
                if self.split:
                    z, z_add = inputs[0], inputs[1]
                    return tf.concat([z, z_add], axis=3)

                else:
                    return inputs

            if self.split:
                z, z_add = inputs[0], inputs[1]
                mean, log_sd = tf.split(self.prior(z), num_or_size_splits=2, axis=3)
                z_sample = mean + tf.math.exp(log_sd) * z_add

                z = tf.concat([z, z_sample], axis=3)
                return z
            else:
                zero = tf.zeros_like(inputs)
                mean, log_sd = tf.split(self.prior(zero), num_or_size_splits=2, axis=3)
                z = mean + tf.math.exp(log_sd) * inputs
                return z


class Conv2D_zeros(Layer):

    def __init__(self, filters):
        super(Conv2D_zeros, self).__init__()
        self.filters = filters

    def build(self, input_shape):
        # Get shape
        b, h, w, c = input_shape

        self.conv = tf.keras.layers.Conv2D(self.filters,
                                           kernel_size=3,
                                           strides=(1, 1),
                                           padding="same",
                                           activation=None,
                                           kernel_initializer="zeros",
                                           bias_initializer="zeros")

        self.scale = self.add_weight(
            shape=(1, 1, 1, self.filters),
            trainable=True,
            initializer="zeros",
            name="conv2D_zeros_scale_" + str(np.random.randint(0, 1e6))
        )

    def call(self, inputs):
        x = self.conv(inputs)
        x = x * tf.math.exp(self.scale * 3)

        return x


def gaussian_logprob(x, mean, log_sd):
    return -0.5 * tf.math.log(2 * np.pi) - log_sd - 0.5 * (x - mean) ** 2 / tf.math.exp(2 * log_sd)


## Step of flow
class FlowStep(Layer):

    def __init__(self, hidden_channels, perm_type="1x1"):
        super(FlowStep, self).__init__()
        self.hidden_channels = hidden_channels

        self.actnorm = ActNorm()
        self.perm = Permutation(perm_type)
        self.coupling = AffineCoupling(hidden_channels)

    def call(self, inputs, forward=True):
        """
        inputs: list containing [input tensor, log_det]
        returns: output tensor
        """

        x = inputs[0]
        log_det = inputs[1]

        if forward:
            x, log_det = self.actnorm([x, log_det], forward)
            x, log_det = self.perm([x, log_det], forward)
            x, log_det = self.coupling([x, log_det], forward)
        else:
            x, log_det = self.coupling([x, log_det], forward)
            x, log_det = self.perm([x, log_det], forward)
            x, log_det = self.actnorm([x, log_det], forward)

        return x, log_det


### Glow model

class Glow(keras.Model):

    def __init__(self, steps, levels, img_shape, hidden_channels, perm_type="1x1"):
        super(Glow, self).__init__()

        assert (len(img_shape) == 3)

        self.steps = steps  # Number of steps in each flow, K in the paper
        self.levels = levels  # Number of levels, L in the paper
        self.height, self.width, self.channels = img_shape
        self.dimension = self.height * self.width * self.channels  # Dimension of input/latent space
        self.hidden_channels = hidden_channels
        self.perm_type = perm_type

        # Normal distribution with 0 mean and std=1, defined over R^dimension
        self.latent_distribution = tf_prob.distributions.MultivariateNormalDiag(
            loc=[0.0] * self.dimension)

        self.squeeze = Squeeze()
        self.flow_layers = []
        self.split_layers = []

        for l in range(levels):
            flows = []

            for k in range(steps):
                flows.append(FlowStep(hidden_channels, perm_type))

            self.flow_layers.append(flows)

            if l != levels - 1:
                self.split_layers.append(Split(split=True))
            else:
                self.split_layers.append(Split(split=False))

    def preprocess(self, x):
        """Expects images x without any pre-processing, with pixel values in [0, 255]"""
        x = x / 255.0 - 0.5  # Normalize to lie in [-0.5, 0.5]
        # Add uniform noise
        x = x + tf.random.uniform(shape=tf.shape(x), minval=0, maxval=1 / 256.0, dtype=x.dtype)

        return x

    def sample_image(self, temperature):

        # Sample latent variable
        z = self.latent_distribution.sample() * temperature
        # Decode z
        x = self(z, forward=False)

        x = np.clip(x, -0.5, 0.5)
        x = x + 0.5
        return x

    def call(self, inputs, forward=True, reconstruct=False):

        if forward:
            x = inputs
            log_det = 0.0
            sum_log_p = 0.0
            x = self.preprocess(x)

            latent_variables = []

            for l in range(self.levels - 1):

                x = self.squeeze(x, forward=forward)

                # K steps of flow
                for k in range(self.steps):
                    x, log_det = self.flow_layers[l][k]([x, log_det], forward=forward)

                # Split
                x, z, log_p = self.split_layers[l](x, forward=forward)
                sum_log_p += log_p

                latent_dim = np.prod(z.shape[1:])  # Dimension of extracted z
                latent_variables.append(tf.reshape(z, [-1, latent_dim]))

            # Last squeeze
            x = self.squeeze(x, forward=forward)

            # Last steps of flow
            for k in range(self.steps):
                x, log_det = self.flow_layers[-1][k]([x, log_det], forward=forward)

            z, log_p = self.split_layers[-1](x, forward=forward)
            sum_log_p += log_p

            latent_dim = np.prod(z.shape[1:])  # Dimension of last latent variable
            latent_variables.append(tf.reshape(z, [-1, latent_dim]))

            # Concatenate latent variables
            latent_variables = tf.concat(latent_variables, axis=1)

            c = -self.dimension * np.log(1 / 256)

            # Average NLL of the images in bits/dimension
            NLL = (-sum_log_p - log_det + c) / (np.log(2) * self.dimension)

            self.add_loss(NLL)

            return latent_variables, NLL

        else:
            # Run the model backwards, assuming that inputs is a sampled latent variable of full dimension

            # Extract slices of the latent variables to be used in reverse split function
            latent_variables = []

            start = 0  # Starting index of the slice for z_i
            stop = 0  # Stopping index of the slice for z_i

            for l in range(self.levels - 1):
                stop += self.dimension // (2 ** (l + 1))
                latent_variables.append(inputs[start:stop])
                start = stop

            latent_variables.append(inputs[start:])

            log_det = 0.0

            # Extract last latent variable and reshape
            z = latent_variables[-1]
            c_last = self.channels * 2 ** (self.levels + 1)  # nr of channels in the last latent output
            h_last = self.height // (2 ** self.levels)  # height of the last latent output
            w_last = self.width // (2 ** self.levels)  # width of the last latent output
            z = tf.reshape(z, shape=(1, h_last, w_last, c_last))

            z = self.split_layers[-1](z, forward=forward, reconstruct=reconstruct)

            # Last steps of flow
            for k in reversed(range(self.steps)):
                z, log_det = self.flow_layers[-1][k]([z, log_det], forward=forward)

            # Last squeeze
            z = self.squeeze(z, forward=forward)

            for l in reversed(range(self.levels - 1)):

                # Extract latent variable, reshape, and concatenate along channel dimension (reverse split)
                z_add = latent_variables[l]
                z_add = tf.reshape(z_add, shape=z.shape)

                z = self.split_layers[l]([z, z_add], forward=forward, reconstruct=reconstruct)

                # K steps of flow
                for k in reversed(range(self.steps)):
                    z, log_det = self.flow_layers[l][k]([z, log_det], forward=forward)

                z = self.squeeze(z, forward=forward)

            x = z

            return x
        
        
        
        
        
print('1111111')
        
        
        
        
        
from matplotlib import pyplot as plt
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
print('imported second batch')

# from model import *


def preprocess(data_set, type=None):
    if type is not None:
        (X_train, Y_train), (X_test, Y_test) = data_set.load_data(type=type)
    else:
        (X_train, Y_train), (X_test, Y_test) = data_set.load_data()
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    X_train = X_train[0:, :, :, :]
    X_test = X_test[0:, :, :, :]

    # Add zero-padding to get 32x32 images
    X_train = np.pad(X_train, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))
    X_test = np.pad(X_test, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    return X_train, Y_train, X_test, Y_test


# Load data sets
X_train, Y_train, X_test, Y_test = preprocess(mnist)
# X_train_letters, Y_train_letters, X_test_letters, Y_test_letters = preprocess(emnist, type='letters')
# X_train_fashion, _, X_test_fashion, _ = preprocess(fashion_mnist)

np.random.shuffle(X_test)
# np.random.shuffle(X_test_letters)
# np.random.shuffle(X_test_fashion)

# Create balanced MNIST training set

num = 4000  # samples per class

X_balanced = np.zeros((num * 10, 32, 32, 1))
count = [0 for i in range(10)]

idx = 0
for i, x in enumerate(X_train):
    if count[Y_train[i]] < num:
        X_balanced[idx] = x
        count[Y_train[i]] += 1
        idx += 1

assert (idx == X_balanced.shape[0])

np.random.shuffle(X_balanced)

print('beginning to train model')
# Train the model
lr = 1e-4

model = Glow(steps=32, levels=3, img_shape=(32, 32, 1), hidden_channels=512, perm_type="1x1")
adam = keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=adam)

print(type(X_balanced))
print((X_balanced.shape))

model.fit(X_balanced, epochs=10, batch_size=128)

# Plot loss
loss = model.history.history["loss"]
epochs = np.arange(len(loss))
plt.plot(epochs, loss, color="r")





print('22222222')






# LINEAR INTERPOLATION

# These are pairs of different examples of the same digit to illustrate the latent space produces smooth transitions
images = [X_test[np.where(Y_test == 0)[0][0:2]],
          X_test[np.where(Y_test == 1)[0][0:2]],
          X_test[np.where(Y_test == 2)[0][0:2]],
          X_test[np.where(Y_test == 3)[0][0:2]],
          X_test[np.where(Y_test == 4)[0][0:2]],
          X_test[np.where(Y_test == 5)[0][0:2]],
          X_test[np.where(Y_test == 6)[0][0:2]],
          X_test[np.where(Y_test == 7)[0][0:2]],
          X_test[np.where(Y_test == 8)[0][0:2]],
          X_test[np.where(Y_test == 9)[0][0:2]]
          ]

zs = []

for pair in images:
    x1, x2 = pair[0:1], pair[1:2]

    # Encode
    z1, _ = model(x1, forward=True)
    z1 = z1[0]
    z2, _ = model(x2, forward=True)
    z2 = z2[0]

    zs.append([z1, z2])

# Interpolate
num = 10
latents_full = []

for pair in zs:
    z1, z2 = pair[0], pair[1]
    lis = []
    alpha = 0
    for i in range(num):
        lis.append((1 - alpha) * z1 + alpha * z2)
        alpha += 1 / (num - 1.0)
    latents_full.append(lis)

# Decode
decodings = []

for i, latents in enumerate(latents_full):
    print(f"decoding {i} of {len(latents_full)}")
    imgs = []
    for i, z in enumerate(latents):
        output = model(z, forward=False, reconstruct=True)
        imgs.append(output)
    decodings.append(imgs)

# Plot
fig = plt.figure(figsize=(8, 8))

for row in range(10):
    for col in range(num):
        fig.add_subplot(10, num, row * num + col + 1)
        plt.imshow(decodings[row][col][0, :, :, 0], cmap="gray")
        plt.axis('off')


        
        
        
print('3333333')

        
        
        
        
        
# Reconstruct one image

plt.figure() # plot input
x_in = X_test[1:2]
plt.imshow(x_in[0])

plt.figure() # plot latent
z, _ = model(x_in, forward=True)
plt.imshow(np.array(z).reshape((32,32)))

plt.figure() # plot outpur
x_out = model(z[0], forward=False, reconstruct=True)
plt.imshow(x_out[0])







# Take 30 samples of output and plot the mean image and std image
samples = []
for i in range(30):
    samples.append(model(z[0], forward=False, reconstruct=False))

plt.figure()
plt.imshow(np.mean(np.array(samples), axis=0)[0])

plt.figure()
plt.imshow(np.std(np.array(samples), axis=0)[0])



print('444444')



np.mean(np.array(samples), axis=0)



print('5555555')




# SAMPLE NEW IMAGES

rows = 10
cols = 10

images = []

for r in range(rows):
    for c in range(cols):
        x = model.sample_image(temperature=4)
        images.append(x)

# Plot
fig = plt.figure(figsize=(8, 8))

for r in range(rows):
    for c in range(cols):
        fig.add_subplot(rows, cols, r * cols + c + 1)
        plt.imshow(images[r * cols + c][0, :, :, 0], cmap="gray")
        plt.axis('off')

        
        
print('66666')
