import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import cv2 #opencv 설치
import time


form_window = uic.loadUiType('./cat_and_dog.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.path = ('../datasets/cat_dog/train/cat.0.jpg', '')
        self.model = load_model('./cat_and_dog0.8268.h5')
        self.pushButton.clicked.connect(self.image_open_slot)

    def image_open_slot(self):
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        flag = True
        while flag:
            ret, frame = capture.read()
            time.sleep(1/60)
            cv2.imwrite('./captrue.png', frame)
            pixmap = QPixmap('./captrue.png')
            self.lbl_image.setPixmap(pixmap)

            try:
                img = Image.open('./captrue.png')
                img = img.convert('RGB')
                img = img.resize((64, 64))
                data = np.asarray(img)
                data = data / 255
                data = data.reshape(1, 64, 64, 3)
            except:
                print('error')
            predict_value = self.model.predict(data)
            if predict_value < 0.5:
                self.lbl_label.setText('고양이일 확률이' + str(((1 - predict_value[0][0])*100).round()) + '% 입니다.')
            else:
                self.lbl_label.setText('강아지 확률이' + str((predict_value[0][0]*100).round()) + '% 입니다.')

            key = cv2.waitKey(33)
            if key == 27:
                flag = False

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())