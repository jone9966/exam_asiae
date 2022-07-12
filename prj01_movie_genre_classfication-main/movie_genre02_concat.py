import pandas as pd
import glob

data_paths = glob.glob('./crawling/genre_crawling/all/movie_genre_*.csv')
print(data_paths)
df = pd.DataFrame()
for data_path in data_paths:
    df_temp = pd.read_csv(data_path)
    df_temp.columns = ['summary', 'genre']
    df = pd.concat([df, df_temp])

df.sort_values(by=['genre'], axis=0, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(df[df['genre'] == 'drama'].index, inplace=True)
df.drop(df[df['genre'] == 'erotic'].index, inplace=True)


print(df.head())
print(df.tail())
print(df['genre'].value_counts())
print(df.info())
df.to_csv('./crawling/movie_genre_year.csv', index=False)
