import pandas as pd

df_trip = pd.read_csv('./crawling_data/reviews_trip_all.csv')
df_naver = pd.read_csv('./crawling_data/reviews_naver_all.csv')

content_all = df_trip['content'].tolist()

area_list = []
content_list = []
review_list = []

for content in content_all:
    area = df_trip[df_trip['content'] == content]['area'].tolist()[0]
    try:
        trip_review = df_trip[df_trip['content'] == content]['reviews'].tolist()[0]
        naver_review = df_naver[df_naver['content'] == content]['reviews'].tolist()[0]
        total_review = trip_review + naver_review
        print(area, content)
    except:
        total_review = df_trip[df_trip['content'] == content]['reviews'].tolist()[0]
        print(area, content)

    area_list.append(area)
    content_list.append(content)
    review_list.append(total_review)

df_concat = pd.DataFrame({'area': area_list, 'content': content_list, 'reviews': review_list})
df_concat.info()
print(df_concat.head())
print(df_concat.tail())
df_concat.to_csv('./crawling_data/reviews_trip_naver.csv', index=False)