from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time


def crawl_summary():
        driver.find_element_by_xpath('//*[@id="content"]/div[1]/div[1]/div[3]/ul/li[{}]/dl/dt/a'.format(i)).click()
        try:
            summary = driver.find_element_by_class_name('con_tx').text
            summary = re.compile('[^가-힣|a-z|A-Z ]').sub(' ', summary)
            text = driver.find_element_by_xpath('//*[@id="content"]/div[1]/div[2]/div[1]/dl').text
            genre = ''
            for l in range(len(genre_kor)):
                if genre_kor[l] in text:
                    genre = genre + genre_eng[l] + ' '
            if len(genre):
                summary_list.append(summary)
                genre_list.append(genre)
            driver.back()
        except NoSuchElementException:
            print('no summary')
            driver.back()


options = webdriver.ChromeOptions()
options.add_argument('lang=ko_KR')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('disable-gpu')
driver = webdriver.Chrome('./chromedriver', options=options)
genre_eng = ['drama', 'fantasy', 'horror', 'romance', 'documentary', 'comedy', 'crime', 'sf', 'action', 'erotic']
genre_kor = ['드라마', '판타지', '공포', '멜로/로맨스', '다큐멘터리', '코미디', '범죄', 'SF', '액션', '에로']

for k in range(1):
    summary_list = []
    genre_list = []
    url = 'https://movie.naver.com/movie/running/premovie.naver#'
    driver.get(url)
    for i in range(1,11):
        try:
            crawl_summary()
        except StaleElementReferenceException:
            driver.get(url)
            time.sleep(1)
            crawl_summary()

    df_section_summary = pd.DataFrame(summary_list, columns=['summary'])
    df_section_summary['genre'] = genre_list
    df_section_summary.to_csv('./crawling/movie_sample.csv', index=False)

driver.close()