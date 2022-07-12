import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections
from konlpy.tag import Okt
from matplotlib import font_manager, rc
import matplotlib as mpl


fontpath = './malgun.ttf'
font = font_manager.FontProperties(fname=fontpath, size=8)
plt.rc('font', family='malgun')
df = pd.read_csv('./crawling_data/cleaned_reviews.csv')

words = df[df['content'] == '명동(서울)']['cleaned_sentences']
words = list(words)
words = words[0].split()

worddict = collections.Counter(words)
worddict = dict(worddict)
wordcloud_img = WordCloud(
    background_color='white', max_words=2000, font_path=fontpath).generate_from_frequencies(worddict)

plt.figure(figsize=(12, 12))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis('off')
plt.show()

