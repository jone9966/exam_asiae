import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization

# Input shape
img_rows = img_cols = 128
channels = 3
img_shape = (img_rows, img_cols, channels)


def generator():
    # Downsampling
    d0 = Input(shape=img_shape)
    d1 = Conv2D(32, kernel_size=4, strides=2, padding='same')(d0)
    d1 = LeakyReLU(alpha=0.2)(d1)
    d1 = InstanceNormalization()(d1)
    d2 = Conv2D(64, kernel_size=4, strides=2, padding='same')(d1)
    d2 = LeakyReLU(alpha=0.2)(d2)
    d2 = InstanceNormalization()(d2)
    d3 = Conv2D(128, kernel_size=4, strides=2, padding='same')(d2)
    d3 = LeakyReLU(alpha=0.2)(d3)
    d3 = InstanceNormalization()(d3)
    d4 = Conv2D(256, kernel_size=4, strides=2, padding='same')(d3)
    d4 = LeakyReLU(alpha=0.2)(d4)
    d4 = InstanceNormalization()(d4)
    # Upsampling
    u1 = UpSampling2D(size=2)(d4)
    u1 = Conv2D(256, kernel_size=4, strides=1, padding='same', activation='relu')(u1)
    u1 = InstanceNormalization()(u1)
    u1 = Concatenate()([u1, d3])
    u2 = UpSampling2D(size=2)(u1)
    u2 = Conv2D(128, kernel_size=4, strides=1, padding='same', activation='relu')(u2)
    u2 = InstanceNormalization()(u2)
    u2 = Concatenate()([u2, d2])
    u3 = UpSampling2D(size=2)(u2)
    u3 = Conv2D(64, kernel_size=4, strides=1, padding='same', activation='relu')(u3)
    u3 = InstanceNormalization()(u3)
    u3 = Concatenate()([u3, d1])
    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
    return Model(d0, output_img)


def discriminator():
    img = Input(shape=img_shape)
    d = Conv2D(64, kernel_size=4, strides=2, padding='same')(img)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, kernel_size=4, strides=2, padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = InstanceNormalization()(d)
    d = Conv2D(256, kernel_size=4, strides=2, padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = InstanceNormalization()(d)
    d = Conv2D(512, kernel_size=4, strides=2, padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = InstanceNormalization()(d)
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d)
    return Model(img, validity)

# Build and compile the discriminators
optimizer = Adam(0.0002, 0.5)
d_A = discriminator()
d_B = discriminator()
d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

# Build the generators
g_AB = generator()
g_BA = generator()

# Input images from both domains
img_A = Input(shape=img_shape)
img_B = Input(shape=img_shape)

# Translate images to the other domain
fake_B = g_AB(img_A)
fake_A = g_BA(img_B)
# Translate images back to original domain
reconstr_A = g_BA(fake_B)
reconstr_B = g_AB(fake_A)
# Identity mapping of images
img_A_id = g_BA(img_A)
img_B_id = g_AB(img_B)

# For the combined model we will only train the generators
d_A.trainable = False
d_B.trainable = False

# Discriminators determines validity of translated images
valid_A = d_A(fake_A)
valid_B = d_B(fake_B)

# Combined model trains generators to fool discriminators
combined = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id])
combined.summary()
combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], loss_weights=[1, 1, 10.0, 10.0, 1.0, 1.0],
                 optimizer=optimizer)


# train
epochs = 600
batch_size = 10
patch = int(img_rows / 2 ** 4)
disc_patch = (patch, patch, 1)
valid = np.ones((batch_size,) + disc_patch)
fake = np.zeros((batch_size,) + disc_patch)

(trainA, testA, trainB, testB) = np.load('./image/image_data.npy', allow_pickle=True)
print(trainA.shape)
print(trainB.shape)
exit()
n_batches = int(min(len(trainA), len(trainB)) / batch_size)
total_samples = n_batches * batch_size

for epoch in range(epochs):
    trainA = np.random.choice(trainA, total_samples, replace=False)
    trainB = np.random.choice(trainB, total_samples, replace=False)
    batch = []
    for i in range(n_batches - 1):
        imgs_A = trainA[i * batch_size:(i + 1) * batch_size]
        imgs_B = trainB[i * batch_size:(i + 1) * batch_size]

        fake_B = g_AB.predict(imgs_A)
        fake_A = g_BA.predict(imgs_B)

        # Train the discriminators (original images = real / translated = Fake)
        dA_loss_real = d_A.train_on_batch(imgs_A, valid)
        dA_loss_fake = d_A.train_on_batch(fake_A, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = d_B.train_on_batch(imgs_B, valid)
        dB_loss_fake = d_B.train_on_batch(fake_B, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss = 0.5 * np.add(dA_loss, dB_loss)

        # Train the generators
        g_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])

    # Plot the progress
    print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f]" % (epoch, epochs, d_loss[0], 100*d_loss[1], g_loss[0]))

    if (epoch+1)%10 == 0 and i == 0:
        sample = np.random.choice(trainA, 1)
        fake_B = g_AB.predict(sample)
        reconstr_A = g_BA.predict(fake_B)

        result = np.concatenate([sample, fake_B, reconstr_A])
        result = 0.5 * result + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(1, 3)
        cnt = 0
        for i in range(3):
            axs[1, i].imshow(result[cnt])
            axs[1, i].set_title(titles[i])
            axs[1, i].axis('off')
            cnt += 1
        fig.savefig("./out_img/{}.jpg".format(epoch))
        plt.close()
g_AB.save('./models/maks.h5')
