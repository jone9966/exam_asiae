import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmwrite, mmread
import pickle
from gensim.models import Word2Vec
import collections


df_reviews = pd.read_csv('./crawling_data/cleaned_reviews.csv')
Tfidf_matrix = mmread('./models/Tfidf_trip_review.mtx').tocsr()
embedding_model = Word2Vec.load('./models/word2VecModel.model')
with open('./models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)


def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True)
    simScore = simScore[1:11]
    contentidx = [i[0] for i in simScore]
    recMovieList = df_reviews.iloc[contentidx]
    return recMovieList

# 제목 검색
words = df_reviews[df_reviews['content'] == '죽성성당(부산)']
content_idx = words.index[0]
print(df_reviews.iloc[content_idx, 1])

words = list(words['cleaned_sentences'])
words = words[0].split()
worddict = collections.Counter(words)
print(worddict.most_common(10))

cosine_sim = linear_kernel(Tfidf_matrix[content_idx], Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)
print(recommendation.iloc[:, 1])



# 키워드 검색
# key_word = '맛집'
# sentence = [key_word] * 11
# sim_word = embedding_model.wv.most_similar(key_word, topn=10)
#
# words = []
# for word, _ in sim_word:
#     words.append(word)
# print(words)
#
# for i, word in enumerate(words):
#     sentence += [word] * (10-i)
# sentence = ' '.join(sentence)
# sentence_vec = Tfidf.transform([sentence])
# cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
# recommendation = getRecommendation(cosine_sim)
# print(recommendation.iloc[:, 1])