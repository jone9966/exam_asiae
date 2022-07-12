from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = './image/face/'
categories = ['no_mask', 'mask']
image_w = image_h = 128

pixel = image_h * image_w * 3
X = []
Y = []
files = None
for idx, category in enumerate(categories):
    imgs = glob.glob(img_dir + category + '/' + category + '*.jpg')
    for i, img in enumerate(imgs):
        try:
            img = Image.open(img)
            img = img.convert('RGB')
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            X.append(data)
            Y.append(idx)
        except:
            print(category, i, '')
X = np.array(X)
Y = np.array(Y)
X = X / 255


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
xy = (X_train, X_test, Y_train, Y_test)
np.save('./image/image_data.npy', xy)