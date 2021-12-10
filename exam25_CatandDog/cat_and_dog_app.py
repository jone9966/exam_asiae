import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

form_window = uic.loadUiType('./cat_and_dog.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.path = ('../datasets/cat_dog/train/cat.0.jpg', '')
        self.model = load_model('./cat_and_dog0.8268.h5')
        self.pushButton.clicked.connect(self.image_open_slot)

    def image_open_slot(self):
        old_path = self.path
        self.path = QFileDialog.getOpenFileName(
            self, 'Open file', '../datasets/cat_dog/train', 'Image Files(*.jpg;*.png);;All Files(*,*)')
        if self.path[0] == '':
            self.path = old_path
        print(self.path)
        pixmap = QPixmap(self.path[0])
        self.lbl_image.setPixmap(pixmap)
        try:
            img = Image.open(self.path[0])
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

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())