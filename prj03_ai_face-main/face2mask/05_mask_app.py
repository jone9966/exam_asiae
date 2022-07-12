import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cvlib as cv
import cv2

form_window = uic.loadUiType('./gui/mask.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.path = ('./image/mask/mask_1.jpg', '')
        self.model = load_model('./models/mask_1.0.h5')
        self.pushButton.clicked.connect(self.image_open_slot)

    def image_open_slot(self):
        old_path = self.path
        self.path = QFileDialog.getOpenFileName(
            self, 'Open file', './image', 'Image Files(*.jpg;*.png);;All Files(*,*)')
        if self.path[0] == '':
            self.path = old_path
        pixmap = QPixmap(self.path[0]).scaledToWidth(256)
        self.lbl_image.setPixmap(pixmap)
        try:
            img = plt.imread(self.path[0])
            face, confidence = cv.detect_face(img, 0.8)
            if not face:
                print('no face')
            f = face[0]
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            img = img[startY:endY, startX:endX, :]
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            predict_value = self.model.predict(img)
            if predict_value < 0.5:
                self.lbl_label.setText('{}% mask off'.format(((1 - predict_value[0][0])*100).round()))
            else:
                self.lbl_label.setText('{}% mask on'.format((predict_value[0][0]*100).round()))
        except:
            self.lbl_label.setText('no face')

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())