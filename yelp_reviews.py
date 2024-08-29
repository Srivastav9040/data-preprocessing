#using random delay for time.sleep()
delays = [7, 4, 6, 2, 10, 19]
delay = np.random.choice(delays)

def get_review(url, res_name, res_address):
    binary = FirefoxBinary('/usr/bin/firefox')
    opts = webdriver.FirefoxOptions()
    opts.add_argument("--headless")
    driver = webdriver.Firefox(firefox_binary=binary, firefox_options=opts )
    driver.get(url)
    
    time.sleep(delay)
    page = driver.page_source
    soup = BeautifulSoup(page, 'lxml')
    review_num = 0
    final_data = []
    num_page = 1
    info_scraped = {}
    info_scraped['reviewer_name'] = None
    #info_scraped['reviewer_stat'] = None
    info_scraped['reviewer_friends'] = None
    info_scraped['reviewer_reviews'] = None
    info_scraped['reviewer_photos'] = None
    info_scraped['ratings'] = None
    info_scraped['comment'] = None
    info_scraped['review_date'] = None
    info_scraped['reviewer_origin'] = None
    info_scraped['reviewer_profile'] = None
    # retrieve the total page number, if there is no information about this, it means the reviews have less than one full page, set page number to 1.
    try:
        total_page = driver.find_element_by_xpath('/html/body/div[2]/div[3]/yelp-react-root/div/div[3]/div/div/div[2]/div/div[1]/div[2]/section[2]/div[2]/div/div[4]/div[2]/span').text
        print(total_page)
        totalpage = [int(s) for s in total_page.split() if s.isdigit()]

        num_page = totalpage[-1]
        print(num_page)
        #num_page = 1

    except:
        print(None)

    # iterate through all pages
    
    print(url)
    for page_np in range(num_page):
        print('[{}] {} scraped page out of {}'.format(datetime.now(), page_np, num_page))
        time.sleep(2)
        page = driver.page_source
        #soup = BeautifulSoup(page, 'lxml')
        soup2 = BeautifulSoup(page, 'lxml')
        
        # retrieve all data on the site
        all = soup.find_all('div', {'class': "main-content-wrap main-content-wrap--full"})
 
        #special_all_stat = soup2.find_all('div',{'class': " margin-t0-5__373c0__1VMSL border-color--default__373c0__3-ifU"})
        special_all_reviews = soup2.find_all('div',{'class': "review__373c0__13kpL border-color--default__373c0__3-ifU"})
        review_num += len(special_all_reviews)

        for i in range(len(special_all_reviews)):
            info_scraped = {}
            default = 'https://www.yelp.com'
            stat = ''
            origin = ''
            # retrieve reviewer name
            try:
                special_user = special_all_reviews[i].find('div',{'class': "user-passport-info border-color--default__373c0__3-ifU"})
                
                info_scraped['reviewer_name'] = special_user.find('a').text
                #print(info_scraped['reviewer_name'])
            except:
                print(None)

            # retrieve reviewer statistic, like number of friends, number of reviews, elite or not.
            try:
                
                for j in special_all_reviews[i].find_all('span', {'class': 'css-1dgkz3l'}) :
                    stat += j.text
                    stat += " "
                #print(stat)
                info_scraped['reviewer_friends'] = stat.split()[0]
                info_scraped['reviewer_reviews'] = stat.split()[1]
                info_scraped['reviewer_photos'] = stat.split()[2]
            except:

                print(None)

            # retrieve the rating of this review
            try:
                
                info_scraped['ratings'] = special_all_reviews[i].find('div', {"aria-label": re.compile('star rating')})["aria-label"].split()[0]
                #print(info_scraped['ratings'])
            except:
                print(None)

            # retrieve the comment text the reviewer left
            try:
                
                info_scraped['comment'] = special_all_reviews[i].find('p', {'class': 'comment__373c0__1M-px css-n6i4z7'}).find('span', {'class': 'raw__373c0__3rcx7'}).text
               
                #print(info_scraped['comment'])
            except:

                print(None)

            # retrieve the date of review
            try:
                info_scraped['review_date'] = datetime.strptime(special_all_reviews[i].find('span', {
                    'class': 'css-e81eai'}).text,'%m/%d/%Y').date()

                #print(info_scraped['review_date'])
            except:
                print(None)

            # retrieve origin of the reviewer and append them to one string
            try:
                origin = special_all_reviews[i].find('span',{'class':'css-n6i4z7'}).text
                 
                info_scraped['reviewer_origin'] = origin
                
            except:
                print(None)

            # retrieve profile website of each reviewer, prepared to retrieve the his history of ratting record
            try:
                info_scraped['reviewer_profile'] = default + special_all_reviews[i].find('a', {'class': 'css-166la90'}).attrs['href']
               
            except:
                print(None)
            print("********************************************************************************")
            
            final_data.append(info_scraped)
            
        #find no of clickable buttons
        clickable_button = soup.find_all('div',{'class': "pagination-link-container__373c0__1mmdE border-color--default__373c0__3-ifU"})
        clicking_links = len(clickable_button)+2
        click_link = str(clicking_links)
        
        # click the next button to go to next page
            
        if page_np == num_page-1:
            break
            
        else:
            driver.find_element_by_xpath(
                '//*[@id="wrap"]/div[3]/yelp-react-root/div/div[3]/div/div/div[2]/div/div[1]/div[2]/section[2]/div[2]/div/div[4]/div[1]/div/div['+click_link+']/span/a/span').click()

    address = res_address.strip()
    restaurant_name = [res_name] * review_num
    address = [address] * review_num

    driver.quit()

    df = pd.DataFrame(final_data)

    df['restaurant name'] = pd.Series(restaurant_name)
    df['address'] = pd.Series(address)
    df.index += 1
    print(df)
    return df


iteration_from = 10
iteration_end = 20
review_data = []

for i in range(iteration_from, iteration_end):
    print(str(i) + " restaurant out of " + str(len(urls)))
    item = urls[i]
    name = res_name[i]
    address = res_add[i]
    resreview = get_review(item, name, address)
    review_data.append(resreview)
    review_all = pd.concat(review_data)
    review_all.to_csv("Reviews"+str(iteration_from)+"-"+str(iteration_end)+".csv")