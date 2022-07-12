import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QStringListModel
from PyQt5.QtGui import QIcon
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle

form_window = uic.loadUiType('./application_recommend_attraction.ui')[0]


class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Triplogger!')
        self.setWindowIcon(QIcon('./image/trip_icon_2.png'))

        # model load
        self.df_reviews = pd.read_csv('./crawling_data/cleaned_reviews.csv')
        self.Tfidf_matrix = mmread('./models/Tfidf_trip_review.mtx').tocsr()
        self.embedding_model = Word2Vec.load('./models/word2VecModel.model')
        with open('./models/tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)

        # images
        self.lbl_list = [self.lbl_0, self.lbl_1, self.lbl_2, self.lbl_4,
                         self.lbl_6,  self.lbl_8, self.lbl_9,
                         self.lbl_11, self.lbl_12, self.lbl_13, self.lbl_14,
                         self.lbl_3, self.lbl_5, self.lbl_7, self.lbl_10, self.lbl_15, self.lbl_16]
        for lbl in self.lbl_list:
            lbl.hide()
        self.lbl_list[0].show()

        # comboBox
        self.areas = list(self.df_reviews['area'].unique())
        self.areas.sort()
        self.cmb_areas.addItem('전체')
        for area in self.areas:
            self.cmb_areas.addItem(area)

        self.contents = list(self.df_reviews['content'])
        self.contents.sort()
        self.cmb_contents.addItem('전체')
        for content in self.contents:
            self.cmb_contents.addItem(content)

        # completer
        model = QStringListModel()
        model.setStringList(self.contents)
        completer = QCompleter()
        completer.setModel(model)
        self.le_keyword.setCompleter(completer)

        # slot connect
        self.cmb_areas.currentIndexChanged.connect(self.cmb_areas_slot)
        self.cmb_contents.currentIndexChanged.connect(self.cmb_contents_slot)
        self.btn_recommend.clicked.connect(self.btn_recommend_slot)

    # slot
    def cmb_areas_slot(self):
        for lbl in self.lbl_list:
            lbl.hide()
        area_num = self.cmb_areas.currentIndex()
        self.lbl_list[area_num].show()

        self.cmb_contents.clear()
        self.cmb_contents.addItem('전체')
        area = self.cmb_areas.currentText()
        if area == '전체':
            for content in self.contents:
                self.cmb_contents.addItem(content)
        else:
            area_contents = list(self.df_reviews[self.df_reviews['area'] == area]['content'])
            area_contents.sort()
            for content in area_contents:
                self.cmb_contents.addItem(content)

    def cmb_contents_slot(self):
        key_word = self.cmb_contents.currentText()
        if key_word:
            if key_word == '전체':
                self.lbl_recommend.setText('')
            else:
                self.setRecommendation(key_word)

    def btn_recommend_slot(self):
        key_word = self.le_keyword.text()
        if key_word:
            try:
                self.setRecommendation(key_word)
            except:
                self.lbl_recommend.setText('제가 모르는 단어입니다.')

    # make recommendation
    def getRecommendation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        contentidx = [i[0] for i in simScore]
        recContentList = self.df_reviews.iloc[contentidx]

        area = self.cmb_areas.currentText()
        if area != '전체':
            recContentList = recContentList[recContentList['area'] == area]
        return recContentList['content']

    def setRecommendation(self, key_word):
        if key_word in self.contents:
            content_idx = self.df_reviews[self.df_reviews['content'] == key_word].index[0]
            cosine_sim = linear_kernel(self.Tfidf_matrix[content_idx], self.Tfidf_matrix)
            recommendation_content = self.getRecommendation(cosine_sim)
            recommendation_content = recommendation_content.iloc[1:11]
        else:
            key_word = key_word.split()
            if len(key_word) < 5:
                key_word = key_word[0]
                sim_word = self.embedding_model.wv.most_similar(key_word, topn=10)
                key_word = [key_word] * 11
                words = []
                for word, _ in sim_word:
                    words.append(word)
                    for i, word in enumerate(words):
                        key_word += [word] * (10 - i)
            sentence = ' '.join(key_word)
            sentence_vec = self.Tfidf.transform([sentence])
            cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrix)
            recommendation_content = self.getRecommendation(cosine_sim)
            recommendation_content = recommendation_content.iloc[:10]

        recommendation_content = '\n'.join(list(recommendation_content))
        self.lbl_recommend.setText(recommendation_content)


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())