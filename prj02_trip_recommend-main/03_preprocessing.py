import pandas as pd
from konlpy.tag import Okt
import re


df = pd.read_csv('./crawling_data/reviews_trip_naver.csv')
df.info()

okt = Okt()

stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)
stopwords_add = pd.read_csv('./crawling_data/stopwords_add.csv', index_col=0)
stopwords = list(stopwords['stopword']) + list(stopwords_add['add_stopword'])
print(stopwords)
count = 0
cleaned_sentences = []
for sentence in df['reviews']:
    count += 1
    if count % 10 == 0:
        print('.', end='')
    if count % 100 == 0:
        print()

    sentence = re.sub('[^가-힣 ]', ' ', sentence)
    token = okt.pos(sentence, stem=True)
    df_token = pd.DataFrame(token, columns=['words', 'class'])
    df_cleaned_token = df_token[(df_token['class'] == 'Noun') |
                                (df_token['class'] == 'Verb') |
                                (df_token['class'] == 'Adjective')]
    words = []
    for word in df_cleaned_token['words']:
        if len(word) > 1:
            if word not in stopwords:
                words.append(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)
df['cleaned_sentences'] = cleaned_sentences
print(df.head())
df = df[['area', 'content', 'cleaned_sentences']]
df.info()
df.to_csv('./crawling_data/cleaned_reviews.csv', index=False)