import glob
import cv2
import numpy as np

img_rows = img_cols = 128
imgset = ['trainA', 'testA', 'trainB', 'testB']
img_pro = []

for data in imgset:
    paths = glob.glob('./image/faceon/{}/*.jpg'.format(data))
    imgs = []
    for path in paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_rows, img_cols))
        img = img.astype(np.float)
        if np.random.random() > 0.5:
            img = np.fliplr(img)
        imgs.append(img)
    imgs = np.array(imgs) / 127.5 - 1.
    img_pro.append(imgs)
np.save('./image/image_data.npy', img_pro)
