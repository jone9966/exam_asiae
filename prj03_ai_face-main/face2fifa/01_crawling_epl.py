from selenium import webdriver
import time
import urllib.request

# 크롬 웹브라우저 실행
options = webdriver.ChromeOptions()
options.add_argument('lang=ko_KR')
options.add_argument('disable_gpu')
driver = webdriver.Chrome('./chromedriver', options=options)
team = [2, 130, 131, 43, 7, 9]
count = 115

for i in team:
    url_list = []
    url = 'https://www.premierleague.com/players?se=418&cl={}'.format(i)
    driver.get(url)
    time.sleep(2)
    for j in range(1, 20):
        try:
            player_xpath = '/html/body/main/div[2]/div/div/div/table/tbody/tr[{}]/td[1]/a'.format(j)
            player = driver.find_element_by_xpath(player_xpath).get_attribute('href')
            url_list.append(player)
        except:
            break

    for url in url_list:
        driver.get(url)
        time.sleep(1)
        imgUrl = driver.find_element_by_xpath('/html/body/main/section/div[2]/div[1]/img').get_attribute("src")
        urllib.request.urlretrieve(imgUrl, './image/faceon/raw/real/player_{}.jpg'.format(count))
        count+=1

driver.close()
