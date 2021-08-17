from selenium import webdriver
import os
from time import sleep
from utils import extract_id
def init_browser():
    global browser
    browser=webdriver.Firefox(executable_path="../geckodriver")
    # browser.get(url)
    sleep(0.5)
    # print('Initial browser...')
    # return browser

def get_product_id_per_page(curr_url,curr_page):
    # browser = init_browser(curr_url)

    list_id = []
    # try:
    browser.get(curr_url)
    sleep(1)
    product_boxes = None
    product_boxes = browser.find_elements_by_xpath("//a[@class='product-item']")
    product_href = None
    product_href = [item.get_attribute('href') for item in product_boxes]
    # print('product_href',product_href)
    for href_ in product_href:
        id_ = extract_id(str(href_))
        if id_ not in list_id and id_.isdigit():
            list_id.append(id_)
        # sleep(0.5)
    # print('list_id',len(list_id))
    # return list_id
    ## save page
    path_product_id = '../../data/product_id'
    folder = os.path.join(path_product_id,'tui-cam-tay-nu')
    if not os.path.isdir(folder):
        os.mkdir(folder)

    path_page = os.path.join(folder,str(curr_page)+'.txt')
    
    if not os.path.isfile(path_page):
        file_save = open(path_page,'w')
        for id_ in list_id:
            file_save.write(id_)
            file_save.write('\n')
        file_save.close()
        print('='*50)
        print('Saved page {}'.format(str(curr_page)))

    return list_id

def get_product_id(url,save_path):

    i = 1
    
    file_save = open(save_path,'a+')

    while True:
        url_page = "{}&page={}&src=static_block".format(url, i)

        try:
            curr_prod_id = get_product_id_per_page(url_page,i)
            if not curr_prod_id:
                break
        except Exception as e:
            print('>'*50)
            print('Fail at page {0} because {1}'.format(str(i),str(e)))
            continue
        i += 1

def get_on_product(curr_url):
    try:
        browser.get(curr_url)

        ## get review
        review_img = browser.find_element_by_xpath("//div[@class='review-images']")
        
        ## get photo/gallery
        gallery = review_img.find_element_by_xpath("//a[@data-view-id='pdp_main_view_gallery' and @class='open-gallery ']")
        if gallery:
            ## open gallery
            gallery.click()
            ## excute slide
            slide = browser.find_elements_by_xpath("//div[@class='slide-item-image']")
            for img in slide:
                img.click()

                ## get img src

                sel_img = browser.find_element_by_xpath("//div[@class='slide-item-image']")
                sel_style = sel_img.get_attribute('style')

                print(sel_style)

                sleep(1)
        else:
            photos = review_img.find_elements_by_xpath("//a[@data-view-id='pdp_main_view_photo']")
            for photo in photos:
                photo.click()
                sleep(1)
            

        
    
    except Exception as e:
        print('Fail {}'.format(e))


if __name__== '__main__':

    init_browser()
    sleep(0.5)
    url = 'https://tiki.vn/tui-cam-tay-nu/c4560?_lc=Vk4wMzkwMDcwMTI%3D'
    save_path = '../../data/product_id/id_backup.txt'
    get_product_id(url,save_path)