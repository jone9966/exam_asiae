import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model

number_GAN_models = []
for i in range(10):
    number_GAN_models.append(load_model('./models/GAN_mnist/generator_mnist_{}.h5'.format(i)))

four_digit_number = 1234
numbers = list(str(four_digit_number))
img = []
for i in numbers:
    i = int(i)
    z = np.random.normal(0, 1, (1, 100))
    fake_imgs = number_GAN_models[i].predict(z)
    fake_imgs = 0.5 * fake_imgs + 0.5
    img.append(fake_imgs)

_, axs = plt.subplots(1, 4, figsize=(5, 20), sharey=True, sharex=True)
cnt = 0
for i in range(4):
    axs[i].imshow(img[i].reshape(28, 28), cmap='gray')
    axs[i].axis('off')
    cnt += 1
plt.show()
plt.close()