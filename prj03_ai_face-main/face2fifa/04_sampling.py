import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization

path = glob.glob('./image/faceon/trainA/*')
batch_images = np.random.choice(path, size=1)
img_rows = img_cols = 256
sample_train = []
img_bf = cv2.imread(batch_images[0])
img_bf = cv2.cvtColor(img_bf, cv2.COLOR_BGR2RGB)
img_bf = cv2.resize(img_bf, (img_rows, img_cols))
img = img_bf.astype(np.float)
sample_train.append(img)

g_AB = load_model('./models/faceon4.h5', custom_objects={'InstanceNormalization': InstanceNormalization})
sample_train = np.array(sample_train) / 127.5 - 1.
fake_train = g_AB.predict(sample_train)

img_af = 0.5 * fake_train + 0.5

fig, axes = plt.subplots(1, 2, figsize=(20,10))
axes[0].set_title('Original')
axes[0].imshow(img_bf)
axes[1].set_title('Translated')
axes[1].imshow(img_af[0])
plt.show()