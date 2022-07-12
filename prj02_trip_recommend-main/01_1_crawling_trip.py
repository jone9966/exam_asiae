from selenium import webdriver
import pandas as pd
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import time
from selenium.webdriver.common.keys import Keys

options = webdriver.ChromeOptions()
options.add_argument('lang=ko_KR')
options.add_argument('disable_gpu')
driver = webdriver.Chrome('./chromedriver', options=options)

area_list = []
content_list = []
reviews_list = []


# 명소 페이지
for i in range(1, 101):
    url = 'https://kr.trip.com/travel-guide/city-100042/tourist-attractions/{}.html'.format(i)
    driver.get(url)
    for j in range(1, 11):
        review_num_xpath = '/html/body/div[1]/div[2]/div/div[2]/div/div[2]/div[5]/div[{}]/a/div[2]/div[1]/span[2]'.format(j)
        try:
            review_num = driver.find_element_by_xpath(review_num_xpath).text.split()
        except:
            continue
        review_num = int(review_num[0].replace(',', ''))
        if review_num < 50:
            continue
        content_title_xpath = '//*[@id="list"]/div[5]/div[{}]/a/div[2]/h3'.format(j)
        content = driver.find_element_by_xpath(content_title_xpath).text
        try:
            driver.find_element_by_xpath(content_title_xpath).click()
        except:
            time.sleep(0.5)
            driver.find_element_by_xpath(content_title_xpath).click()
        area = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/div/div/div[2]/nav/div[5]/a').text

        # 리뷰 페이지
        for k in range(1, 100):
            time.sleep(0.2)
            try:
                review = driver.find_element_by_class_name("gl-poi-detail_comment-list").text
            except:
                print('no review')
                break
            review = review.replace('\n', ' ')
            if review:
                area_list.append(area)
                content_list.append(content)
                reviews_list.append(review)
                print(area, content, review)

            try:
                driver.find_element_by_class_name("btn-next.disabled")
                print('no more review')
                break
            except:
                try:
                    driver.find_element_by_class_name("btn-next").send_keys(Keys.ENTER)
                except:
                    print('no more review')
                    break

        driver.back()

    df_review = pd.DataFrame({'area': area_list, 'content': content_list, 'reviews': reviews_list})
    df_review.drop_duplicates(inplace=True)
    df_review.to_csv('./crawling_data/trip/reviews_trip_{}.csv'.format(i), index=False)
    area_list = []
    content_list = []
    reviews_list = []
driver.close()
