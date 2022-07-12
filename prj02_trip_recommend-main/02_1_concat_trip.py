import pandas as pd
import glob


data_paths = glob.glob('./crawling_data/trip/reviews_trip_*.csv')
df = pd.DataFrame()
for data_path in data_paths:
    df_temp = pd.read_csv(data_path)
    df_temp.columns = ['area', 'content', 'reviews']
    df = pd.concat([df, df_temp], ignore_index=True)
df.dropna(inplace=True)
df.drop_duplicates(subset='content', inplace=True)
df.sort_values(by=['area'], axis=0, inplace=True)
print(df.head())

area_list = []
content_list = []
one_sentences = []
for area in df['area'].unique():
    area_unique = df[df['area'] == area]
    for content in area_unique['content'].unique():
        temp = area_unique[area_unique['content'] == content]
        temp = temp['reviews']
        one_sentence = ' '.join(temp)
        content = content + '(' + area + ')'
        stopwrods = ['작성일', '원문보기', '더 보기', '더보기', '완벽해요', '최고에요', '보통이에요', '최악이에요', '모두 보기', '좋아요']
        for word in stopwrods:
            one_sentence = one_sentence.replace(word, ' ')
        area_list.append(area)
        content_list.append(content)
        one_sentences.append(one_sentence)


df = pd.DataFrame({'area': area_list, 'content': content_list, 'reviews': one_sentences})
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(df.head())


# 명소 띄어쓰기 제거
content_list = []
for content in df['content']:
    content = ''.join(content.split())
    content_list.append(content)
df_temp = pd.DataFrame({'area': df['area'], 'content': content_list, 'reviews': df['reviews']})
print(df_temp.head())

# 중복 값 합치기
idx_list = []
for content in content_list:
    if content_list.count(content) > 1:
        idx = df_temp[df_temp['content'] == content].index
        idx_list = idx_list + idx.tolist()
idx_list = list(set(idx_list))
df_temp = df_temp.loc[idx_list]
df_temp.drop_duplicates(subset='content', inplace=True)
print(df_temp.head())


# 띄어쓰기 복구
df.drop(index=idx_list, axis=0, inplace=True)
df = pd.concat((df, df_temp))
df.sort_values(by=['area', 'content'], axis=0, inplace=True)
print(df.head())


# 저장
df.to_csv('./crawling_data/reviews_trip_all.csv', index=False)