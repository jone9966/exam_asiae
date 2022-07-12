import copy
import sys
import time

from PyQt5.QtWidgets import *
from PyQt5 import uic # ui를 클래스로 바꿔준다.
import pandas as pd
import numpy as np
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
import pickle

# ====================gpu 사용 안하려면========================
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
# ============================================================

form_window = uic.loadUiType('./GUI/movie_genre.ui')[0]

class Exam(QMainWindow, form_window):
    def __init__(self):  # 버튼 누르는 함수 처리해 주는 곳
        super().__init__()
        self.setupUi(self)
        self.model = load_model('./models/movie_classfication_model_0.4605516493320465.h5')
        self.stopwords = pd.read_csv('./crawling/stopwords.csv', index_col=0)
        # 단어를 숫자에 대응
        with open('./models/movie_token_2000.pickle', 'rb') as f:
            self.token = pickle.load(f)
        with open('./models/movie_encoder_2000.pickle', 'rb') as f:
            self.encoder = pickle.load(f)
        self.label = self.encoder.classes_
        self.btn_process.clicked.connect(self.process_clicked)
        self.btn_clear.clicked.connect(self.clear_clicked)
        self.lbl_predict_title = [self.lbl_predict_1, self.lbl_predict_2, self.lbl_predict_3]
        self.action_Undo.triggered.connect(self.txt_summary.undo)
        self.action_Redo.triggered.connect(self.txt_summary.redo)

    def clear_clicked(self):
        self.txt_summary.setText('')
        for lbl in self.lbl_predict_title:
            lbl.setText('')

    def process_clicked(self):
        input_str = self.txt_summary.toPlainText()
        self.clear_clicked()
        self.txt_summary.setText(input_str)
        if input_str == '': self.lbl_predict_1.setText('내용을 입력하세요')
        else:
            result_str = self.judge_input(input_str)
            for i in range(len(result_str)):
                self.lbl_predict_title[i].setText(result_str[i])

    def judge_input(self, input_str):
        okt = Okt()
        input_str = okt.morphs(input_str, stem=True)

        words = []
        for i in range(len(input_str)):
            if len(input_str[i]) > 1 and input_str[i] not in list(self.stopwords['stopword']):
                words.append(input_str[i])
        input_str = ' '.join(words)
        print(input_str)

        Max = 2000
        tokened_X = self.token.texts_to_sequences([input_str])
        if Max < len(tokened_X):
            tokened_X = tokened_X[:Max]

        X_pad = pad_sequences(tokened_X, Max)  # 최대 길이에 맞게 늘려준다.
        pred = self.model.predict(X_pad)

        pred = pred[0].tolist()
        list_temp = copy.deepcopy(pred)
        list_temp.sort()
        list_temp.reverse()  # 리스트 정렬
        sum_accuracy, coun = 0, 0
        list_genre = []
        for i in list_temp:
            sum_accuracy += i
            coun += 1
            list_genre.append(self.label[pred.index(i)] + ' - ' + str(round(i * 100, 2)) + '%')
            if sum_accuracy >= 0.7 or coun == 3:
                break
        return list_genre


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())