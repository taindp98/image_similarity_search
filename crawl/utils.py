import re
# import requests
import urllib.request

def extract_url(url):
    re_pattern = r'\".*?\"'
    url_extr = re.findall(re_pattern,url)[0].replace(r'"',r'')
    return url_extr

def extract_id(href_):
    # re_pattern = r'\".*?\"'
    # url_extr = re.findall(re_pattern,url)[0].replace(r'"',r'')
    # return url_extr
    original_link = href_.split('?')[0].replace('.html','')
    id_ = original_link.split('-')[-1][1:]
    return id_

def get_html(url):
    response = urllib.request.urlopen(urllib.request.Request(url))
    return response.read()


# print(get_html('https://tiki.vn/tui-cam-tay-nu/c4560?page=10&src=Vk4wMzkwMDcwMTI%3D'))