from selenium import webdriver
import time
import pandas as pd
import glob

# 크롬 웹브라우저 실행
options = webdriver.ChromeOptions()
options.add_argument('lang=ko_KR')
options.add_argument('disable_gpu')
driver = webdriver.Chrome('./chromedriver', options=options)

df_reviews = pd.read_csv('./crawling_data/reviews_trip_all.csv')
df_reviews.info()
area = df_reviews['area'].tolist()
content = df_reviews['content'].tolist()
area_list = []
content_list = []
reviews_list = []

for i in range(80,233):
    url_list = []
    for j in range(1, 4):  # 블로그 페이지
        url = 'https://section.blog.naver.com/Search/Post.nhn?pageNo={}&rangeType=ALL&orderBy=sim&keyword={}'.format(
            j, content[i])
        driver.get(url)
        time.sleep(0.5)
        for k in range(1, 8):  # 각 블로그 주소 저장
            try:
                title_xpath = '/html/body/ui-view/div/main/div/div/section/div[2]/div[{}]/div/div[1]/div[1]/a[1]'.format(
                    k)
                title = driver.find_element_by_xpath(title_xpath).get_attribute('href')
                url_list.append(title)
            except:
                break

    reviews = ''
    for url in url_list:
        driver.get(url)
        driver.switch_to.frame('mainFrame')
        overlays = ".se-component.se-text.se-l-default"  # 내용 크롤링
        review = driver.find_elements_by_css_selector(overlays)
        review2 = driver.find_elements_by_css_selector('.post-view')
        review = review + review2
        for r in review:
            r = r.text.replace('\n', ' ')
            reviews = reviews + r
    print(content[i], reviews)
    area_list.append(area[i])
    content_list.append(content[i])
    reviews_list.append(reviews)

    if (i+1) % 10 == 0:
        df_review = pd.DataFrame({'area': area_list, 'content': content_list, 'reviews': reviews_list})
        print(df_review)
        df_review.to_csv('./crawling_data/naver/reviews_naver_{}.csv'.format(i), index=False)
        area_list = []
        content_list = []
        reviews_list = []

df_review = pd.DataFrame({'area': area_list, 'content': content_list, 'reviews': reviews_list})
print(df_review)
df_review.to_csv('./crawling_data/naver/reviews_naver_remain.csv', index=False)
driver.close()

data_paths = glob.glob('./crawling_data/naver/reviews_naver_*.csv')
df_naver = pd.DataFrame()
for data_path in data_paths:
    df_temp = pd.read_csv(data_path)
    df_temp.columns = ['area', 'content', 'reviews']
    df_naver = pd.concat([df_naver, df_temp], ignore_index=True)
df_naver.dropna(inplace=True)
df_naver.drop_duplicates(subset='content', inplace=True)
df_naver.sort_values(by=['area', 'content'], axis=0, inplace=True)
df_naver.info()
df_naver.to_csv('./crawling_data/reviews_naver_all.csv', index=False)