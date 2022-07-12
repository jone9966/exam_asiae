from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import os


options = webdriver.ChromeOptions()
options.add_argument('lang=ko_KR')
options.add_argument('disable_gpu')
driver = webdriver.Chrome('./chromedriver', options=options)

dir = "./image/son2mask/son/mask/"
keyword = '진 마스크'
url = "https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl"

driver.get(url)
search = driver.find_element_by_name("q")
search.send_keys(keyword)
search.send_keys(Keys.RETURN)

# 스크롤 조절
SCROLL_PAUSE_TIME = 1
# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")  # 브라우저의 높이를 자바스크립트로 찾음
while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 브라우저 끝까지 스크롤을 내림
    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)
    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try:
            driver.find_element_by_css_selector(".mye4qd").click()
        except:
            break
    last_height = new_height

imgs = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
count = 1
for img in imgs:
    try:
        img.click()
        time.sleep(0.5)
        imgUrl = driver.find_element_by_xpath(
            '//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img').get_attribute("src")
        urllib.request.urlretrieve(imgUrl, dir+'jmask_{}.jpg'.format(count))
        count = count + 1
        if count >= 100:
            break
    except:
        pass
driver.close()





# def createDirectory(directory):
#     try:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     except OSError:
#         print("Error: Failed to create the directory.")