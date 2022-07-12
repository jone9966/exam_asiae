import pandas as pd
import numpy as np

from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model

pd.set_option('display.unicode.east_asian_width', True)
# data load
df = pd.read_csv('./crawling/movie_sample.csv')
X = df['summary']
Y = df['genre']

# target labeling
with open('./models/multi/movie_encoder_year.pickle', 'rb') as f:
    encoder = pickle.load(f)
labeled_Y = []
label = encoder.classes_
for y in Y:
    y = y.split()
    labeled_Y.append(encoder.transform(y))

onehot_Y = []
for y in labeled_Y:
    y = to_categorical(y, 10).tolist()
    onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for onehot_y in y:
        for i in range(10):
            onehot[i] = onehot[i] + onehot_y[i]
    onehot_Y.append(onehot)
onehot_Y = np.array(onehot_Y)

print(onehot_Y[:5])

# 형태소 분리, 한 글자/불용어 제거
okt = Okt()
stopwords = pd.read_csv('./crawling/stopwords.csv', index_col=0)
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
    words = []
    for word in X[i]:
        if len(word) > 1:
            if word not in list(stopwords['stopword']):
                words.append(word)
    X[i] = ' '.join(words)
print(X[:5])


# titles tokenizing
with open('./models/multi/movie_token_year.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)
print(tokened_X[:5])

# padding
X_pad = pad_sequences(tokened_X, 369)
print(X_pad[:5])


model = load_model('./models/multi/movie_genre_classification_model_0.4015904664993286.h5')
pred = model.predict(X_pad)
sample = np.random.randint(8)
print('pred is ', pred[sample])
print('actual is ', onehot_Y[sample])
print('Target :', label[np.where(onehot_Y[sample]==1)])
print('Prediction after learning is ', label[np.where(pred[sample]>0.3)])


