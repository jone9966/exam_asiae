import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
pd.set_option('display.unicode.east_asian_width', True)

df = pd.read_csv('./crawling/movie_genre_year.csv')
print(df.head())
print(df.info())
X = df['summary']
Y = df['genre']

encoder = LabelEncoder()
encoder.fit(['drama', 'fantasy', 'horror', 'romance', 'documentary', 'comedy', 'crime', 'sf', 'action', 'erotic'])
label = encoder.classes_
labeled_Y = []
for y in Y:
    y = y.split()
    labeled_Y.append(encoder.transform(y))
with open('./models/multi/movie_encoder_del.pickle', 'wb') as f:
    pickle.dump(encoder, f)


onehot_Y = []
for y in labeled_Y:
    y = to_categorical(y, 10).tolist()
    onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for onehot_y in y:
        for i in range(10):
            onehot[i] = onehot[i] + onehot_y[i]
    onehot_Y.append(onehot)
onehot_Y = np.array(onehot_Y)
print(onehot_Y)

stopwords = pd.read_csv('./crawling/stopwords.csv', index_col=0)
okt = Okt()
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
    words = []
    for word in X[i]:
        if len(word) > 1:
            if word not in list(stopwords['stopword']):
                words.append(word)
    X[i] = ' '.join(words)
    print(X[i])

token = Tokenizer()
token.fit_on_texts(X)
tokened_X = token.texts_to_sequences(X)
with open('./models/multi/movie_token_del.pickle', 'wb') as f:
    pickle.dump(token, f)
print(tokened_X[:10])

# max = 0
# for words in tokened_X:
#     if max < len(words):
#         max = len(words)
# print(max)

wordsize = len(token.word_index) + 1
print(wordsize)

for i in range(len(tokened_X)):
    if 500 < len(tokened_X[i]):
        tokened_X[i] = tokened_X[i][:500]


X_pad = pad_sequences(tokened_X, 500)
print(X_pad[:10])

X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)
print(X_train[:5], Y_train[:5])
print(X_test[:5], Y_test[:5])

xy = X_train, X_test, Y_train, Y_test
np.save('./models/multi/movie_data_max_{}_wordsize_{}'.format(500, wordsize), xy)

