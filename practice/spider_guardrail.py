#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os
import requests
import urllib
import itertools

url = 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1513214430057_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E9%A9%AC%E8%B7%AF%E3%80%80%E6%8A%A4%E6%A0%8F'
imgs_dir = '/home/wzj/Pictures/spider_guardrail'

if not os.path.exists(imgs_dir):
    os.mkdir(imgs_dir)


def download_img(img_url, img_dir, filename):
    print('download %d :' % filename, img_url)
    filename = os.path.join(img_dir, str(filename) + '.jpg')

    try:
        res = requests.get(img_url, timeout=15)
        if str(res.status_code)[0] == '4':
            print(res.status_code, img_url)
            return False
    except Exception as e:
        print(e)
        print('download failed')
        return False

    with open(filename, 'wb') as f:
        f.write(res.content)
    return True


def build_urls(word):
    print('origin:', word)
    word = urllib.parse.quote(word)
    print('after:', word)
    url = r"http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&st=-1&ic=0&word={word}&face=0&istype=2nc=1&pn={pn}&rn=60"
    urls = (url.format(word=word, pn=x) for x in itertools.count(start=0, step=60))
    return urls


str_table = {
    '_z2C$q': ':',
    '_z&e3B': '.',
    'AzdH3F': '/'
}

char_table = {
    'w': 'a',
    'k': 'b',
    'v': 'c',
    '1': 'd',
    'j': 'e',
    'u': 'f',
    '2': 'g',
    'i': 'h',
    't': 'i',
    '3': 'j',
    'h': 'k',
    's': 'l',
    '4': 'm',
    'g': 'n',
    '5': 'o',
    'r': 'p',
    'q': 'q',
    '6': 'r',
    'f': 's',
    'p': 't',
    '7': 'u',
    'e': 'v',
    'o': 'w',
    '8': '1',
    'd': '2',
    'n': '3',
    '9': '4',
    'c': '5',
    'm': '6',
    '0': '7',
    'b': '8',
    'l': '9',
    'a': '0'
}

char_table = {ord(key): ord(value) for key, value in char_table.items()}


def decode(url):
    for key, value in str_table.items():
        url = url.replace(key, value)
    url = url.translate(char_table)
    return url


def resolve_img_urls(html):
    re_url = re.compile(r'"objURL":"(.*?)"')
    img_urls = [decode(x) for x in re_url.findall(html)]
    return img_urls


if __name__ == '__main__':

    word = '马路　护栏'
    urls = build_urls(word)
    index = 1

    for url in urls:
        if index > 1000:
            break
        html = requests.get(url, timeout=10).content.decode('utf-8')
        img_urls = resolve_img_urls(html)
        for img_url in img_urls:
            is_download = download_img(img_url, imgs_dir, index)
            if is_download:
                index += 1
