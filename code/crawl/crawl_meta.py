import os
import csv
import json
import numpy as np
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup
from collections import OrderedDict

from urllib.parse import urljoin
from urllib.request import urlretrieve
from glob import glob

product_url = "https://tiki.vn/api/v2/products/{}"

flatten_field = ["badges", "inventory", "categories", "rating_summary", "brand",
                 "seller_specifications", "current_seller", "other_sellers",
                 "configurable_options", "configurable_products", "specifications",
                 "product_links", "services_and_promotions", "promotions", "stock_item",
                 "installment_info"]

order_list = ["id", "sku", "name", "url_key", "url_path", "type", "book_cover",
              "short_description", "price", "list_price", "price_usd", "badges",
              "discount", "discount_rate", "rating_average", "review_count", "order_count",
              "favourite_count", "thumbnail_url", "has_ebook", "inventory_status",
              "is_visible", "productset_group_name", "is_fresh", "seller", "is_flower",
              "is_gift_card", "inventory", "url_attendant_input_form", "master_id",
              "salable_type", "data_version", "categories", "meta_title",
              "meta_description", "meta_keywords", "liked", "rating_summary",
              "description", "return_policy", "warranty_policy", "brand",
              "seller_specifications", "current_seller", "other_sellers",
              "configurable_options", "configurable_products", "specifications",
              "product_links", "services_and_promotions", "promotions", "stock_item",
              "installment_info", "video_url", "youtube", "is_seller_in_chat_whitelist",
              "last_update","images"]


def format_product_meta(product):
    product = json.loads(product)
    if not product.get("id", False):
        return None

    format_product = {}

    for field in order_list:
        if field in product:
            if field in flatten_field:
                format_product[field] = json.dumps(
                    product[field], ensure_ascii=False)
            else:
                format_product[field] = product[field]
        else:
            format_product[field] = np.nan
    
    format_product["extra_feature"] = []
    for field in product:
        if field not in order_list:
            extra = "{}: {}".format(field, json.dumps(product[field], ensure_ascii=False))
            format_product["extra_feature"].append(extra)

    return format_product


def crawl_one_product(product_id):
    headers = {"user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36"}

    response = requests.get(product_url.format(product_id),headers=headers)
    if (response.status_code == 200):
        product_meta = format_product_meta(response.text)
        return product_meta


def crawl_list_products(category_file):
    with open(category_file, "r+") as f:
        products_list = list(f)

    product_meta_list = []
    for product_id in products_list:
        # print(product_id)
        product_meta_list.append(crawl_one_product(product_id.strip("\r\n")))

    return product_meta_list


def save_product_list(output_file, product_meta_list):
    file = open(output_file, "w")
    csv_writer = csv.writer(file)

    count = 0

    print("Save file to ", output_file)
    for p in product_meta_list:
        if p is not None:
            if count == 0:
                header = p.keys()
                csv_writer.writerow(header)
                count += 1
            csv_writer.writerow(p.values())
    file.close()
    
    print("="*60)


if __name__ == "__main__":
    # category_file = "../../data/category_list.txt"
    # with open(category_file, "r+") as f:
        # urls_list = list(f)

    # for url in urls_list[0:1]:
        # url = url.strip("\r\n")
    url = 'https://tiki.vn/tui-cam-tay-nu/c4560?_lc=Vk4wMzkwMDcwMTI%3D'
    output_file = "../../data/product_meta/{}.csv".format(url.split("/")[3])

    list_final_prod_meta = []

    if not os.path.exists(output_file):
        input_folder = "../../data/product_id/{}".format(url.split("/")[3])
        # print('input_folder',input_folder)
        input_files = glob(os.path.join(input_folder,'*.txt'))
        for f in tqdm(input_files):
            list_final_prod_meta += crawl_list_products(f)
        save_product_list(output_file, list_final_prod_meta)
