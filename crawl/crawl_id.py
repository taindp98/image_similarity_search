import os
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
from urllib.request import urlretrieve
from utils import extract_id
import urllib
# HOST = "https://tiki.vn"
# base_url = urljoin(
#     # HOST, "thoi-trang/c914?src=c.914.hamburger_menu_fly_out_banner")
#     HOST, 'tui-xach-cong-so-nam/c5337?src=c.914.hamburger_menu_fly_out_banner')

headers = {"user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36"}

def crawl_category_urls(base_url):
    category_list = set()
    category_list.add(base_url)
    tmp_list = [base_url]

    while tmp_list:
        url = tmp_list[0]
        response = requests.get(url)

        tmp_list.remove(url)

        parser = BeautifulSoup(response.text, "html.parser")
        urls_list = parser.select(
            "div.list-group-item.is-child > a.list-group-item")

        for url_ in urls_list:
            url_ = urljoin(HOST, url_.get("href"))
            # print('url',url_)
            tmp_list.append(url_)
            category_list.add(url_)

    return category_list


def crawl_product_id(url):
    product_list = []
    i = 1
    file_path = '/home/taindp/computer_vision/image_retrieval/data/product_id/id_backup.txt'
    
    wf = open(file_path, "a+")
    # with tqdm(total=total_page) as pbar:
    while i!=3:
        # print("Crawl page: ", i)
        
        url_page = "{}&page={}&src=static_block".format(url, i)
        print('url_page',url_page)
        try:
            product_box = None
            # response = urllib.request.urlopen(urllib.request.Request(url_page))
            response = requests.get(url_page,headers=headers)
            # response = requests.get("https://tiki.vn/tui-cam-tay-nu/c4560?page=10&src=Vk4wMzkwMDcwMTI%3D")
            # print(response)
            parser = BeautifulSoup(response.text, "html.parser")

            check_url_products = parser.findAll(type='application/ld+json')
            print('check_url_products',check_url_products)
            product_box = parser.findAll(class_="product-item")
            if (len(product_box) == 0):
                break
            # print(len(product_box))
            for product in product_box:
                href_ = product.get('href')
                print('href',href_)
                id_ = extract_id(href_)
                if id_ not in product_list:
                    product_list.append(id_)
                # if id_ not in wf.readlines():
                    ## write immediate
                    wf.write(id_)
                    wf.write('\n')
            i += 1
            # pbar.update(1)
            print('Amount Pages: {}'.format(str(i)))
            wf.close()
        except Exception as e:
            print('Fail {0} at Page {1}'.format(str(e),str(i)))
            i += 1
            wf.close()
            continue

    return product_list, i


def write_to_txt(file_path, data_list):
    if os.path.exists(file_path):
        return

    with open(file_path, "w+") as wf:
        string = "\n".join(data_list)
        wf.write(string)

    print("Save file to ", file_path)


if __name__ == "__main__":
    category_file = "../../data/category_list.txt"

    # if not os.path.exists(category_file):
        # crawl_category_urls(base_url)

    # else:
    # with open(category_file, "r+") as f:
        # urls_list = list(f)

    # for url in urls_list[0:1]:
        # url = url.strip("\r\n")
        # print(url)
        # output_path = "../../data/product_id/{}.txt".format(url.split("/")[3])
        # if not os.path.exists(output_path):
        # print("Url: ", url)
        # total_page = 50000
        # try:
    url = 'https://tiki.vn/tui-cam-tay-nu/c4560?_lc=Vk4wMzkwMDcwMTI%3D'
    product_list, page = crawl_product_id(url)
        # print('product_list',product_list)
        # print("No. Page: ", page)
        # print("No. Product ID: ", len(product_list))
        # write_to_txt(output_path, product_list)
        # print("="*60)
        # except:
        # write_to_txt(output_path, product_list)
