from selenium import webdriver
from selenium.webdriver.common.proxy import Proxy, ProxyType

boston_res_info = pd.read_csv("combined_reviews_info13001.csv", encoding='utf-8-sig')
rev_urls = boston_res_info['reviewer_profile'].tolist()

delays = [1,2,3]
delay = np.random.choice(delays)

#keep changing proxy list based on https://sslproxies.org/
def get_review(url):

    myProxy = ['167.172.123.221','129.21.253.179','159.89.221.73']
    proxys = np.random.choice(myProxy)
    proxy = Proxy({
    'proxyType': ProxyType.MANUAL,
    'httpProxy': proxys,
    'ftpProxy': proxys,
    'sslProxy': proxys,
    })

    opts = webdriver.FirefoxOptions()
    opts.add_argument("--headless")
    
    driver = webdriver.Firefox(firefox_options=opts,proxy=proxy,executable_path='C:/Users/user/Downloads/geckodriver.exe')
    driver.get(url)
    time.sleep(delay)

    print(url)
    page = driver.page_source
    soup = BeautifulSoup(page, 'lxml')

    info_scraped = {}
    final_data = []

    # retrieve the total page number
    try:
        info_scraped['Name'] = soup.find('h1').text
    except:
        print(None)

    try:
        info_scraped['Origin'] = soup.find('h3', {'class': 'user-location alternate'}).text
    except:
        print(None)

    time.sleep(delay)
    # retrieve the number of rating history recorde
    try:
        rating_list = soup.find_all('td', {'class': 'histogram_count'})
        info_scraped['5_star'] = rating_list[0].text
        info_scraped['4_star'] = rating_list[1].text
        info_scraped['3_star'] = rating_list[2].text
        info_scraped['2_star'] = rating_list[3].text
        info_scraped['1_star'] = rating_list[4].text
    except:
        print(None)

    final_data.append(info_scraped)
    df = pd.DataFrame(final_data)
    df.index += 1

    driver.quit()
    return df


iteration_from = 500
iteration_end = 1000
review_data = []
for i in range(iteration_from, iteration_end):
    print(str(datetime.now()) + ' ' + str(i) + " reviewer out of " + str(len(rev_urls)))
    item = rev_urls[i]

    reviewer_info = get_review(item)
    review_data.append(reviewer_info)
    review_star = pd.concat(review_data)
    review_star.to_csv("Rating distribution-13001-"+str(iteration_from)+"-"+str(iteration_end)+".csv", encoding='utf-8-sig')