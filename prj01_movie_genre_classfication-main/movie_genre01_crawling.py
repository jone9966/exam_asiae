from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time

def crawl_summary():
    text = driver.find_element_by_xpath('//*[@id="old_content"]/ul/li[{}]/ul'.format(i)).text
    for l in range(len(genre_kor)):
        if genre_kor[l] in text:
            driver.find_element_by_xpath('//*[@id="old_content"]/ul/li[{}]/a'.format(i)).click()
            try:
                summary = driver.find_element_by_class_name('con_tx').text
                summary = re.compile('[^가-힣|a-z|A-Z ]').sub(' ', summary)
                summary_list.append(summary)
                genre_list.append(genre_eng[l])
                print(summary)
                print(genre_eng[l])
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
genre_eng = ['fantasy', 'horror', 'romance', 'documentary', 'comedy', 'crime', 'sf', 'action']
genre_kor = ['판타지', '공포', '로맨스', '다큐멘터리', '코미디', '범죄', 'SF', '액션']
pages = [40,35,40,45,50,55,70,75,60,40,40,25,20,25,25,20,20,35,15,15,15,15]
year = 2021

for page in pages:
    summary_list = []
    genre_list = []
    for j in range(1, page):
        url = 'https://movie.naver.com/movie/sdb/browsing/bmovie.naver?open={}&page={}'.format(year,j)
        driver.get(url)
        for i in range(1,20):
            try:
                crawl_summary()
            except StaleElementReferenceException:
                driver.get(url)
                time.sleep(1)
                crawl_summary()
        print(len(summary_list))
        print(len(genre_list))

    df_section_summary = pd.DataFrame(summary_list, columns=['summary'])
    df_section_summary['genre'] = genre_list
    df_section_summary.to_csv('./crawling/genre_crawling/movie_genre_{}.csv'.format(year), index=False)
    year = year - 1

driver.close()