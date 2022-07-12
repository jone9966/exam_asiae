import glob
import dlib
import matplotlib.pyplot as plt
import numpy as np
import cv2


imgs = glob.glob('./image/faceon/raw/real/*.jpg')
for i, img in enumerate(imgs):
    img = plt.imread(img)
    face_in_img = img[0:250, 150:350, :]
    plt.imsave('./image/real/real_{}.jpg'.format(i), face_in_img)