import sys, os, cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization

form_window = uic.loadUiType('./gui/fifa.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.img_rows = self.img_cols = 256
        self.path = ('./image/faceon/testA/real_0.jpg', '')
        self.model = load_model('./models/faceon4.h5', custom_objects={'InstanceNormalization': InstanceNormalization})
        self.pushButton.clicked.connect(self.image_open_slot)

    def image_open_slot(self):
        old_path = self.path
        self.path = QFileDialog.getOpenFileName(
            self, 'Open file', './image', 'Image Files(*.jpg;*.png);;All Files(*,*)')
        if self.path[0] == '':
            self.path = old_path
        image_bf = QPixmap(self.path[0]).scaled(self.img_rows, self.img_cols)
        self.lbl_bf.setPixmap(image_bf)
        try:
            img_bf = plt.imread(self.path[0])[:, :, :3]
            img_bf = cv2.resize(img_bf, (self.img_rows, self.img_cols))
            img_bf = [img_bf.astype(np.float)]
            img_bf = np.array(img_bf) / 127.5 - 1.
            image_af = self.model.predict(img_bf)
            image_af = 0.5 * image_af + 0.5
            plt.imsave('result.jpg', image_af[0])
            image_af = QPixmap('result.jpg')
            self.lbl_af.setPixmap(image_af)
            os.remove('result.jpg')
        except:
            self.lbl_af.setText('no face')

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())