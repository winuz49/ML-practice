#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from ConfigParser import ConfigParser
import requests
import time
import logging


def Request(url=None, data=None, method=None, json=True):
    if not method:
        method = 'post' if data else 'get'

    logger = logging.getLogger('Request')
    for i in xrange(3):
        try:
            resp = requests.request(method=method, url=url, data=data, timeout=60)
            if resp.status_code == 200:
                if not json:
                    return resp
                elif 'error_message' not in resp.json():
                    return resp
            else:
                logger.error('%s %s', url, resp.text)
        except Exception as e:
            logger.error('%s %s', url, e)
        time.sleep(3)


class MyConfigParser(ConfigParser):

    '''
    使配置文件中key可有大写
    '''

    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults)

    def optionxform(self, optionstr):
        return optionstr


class Dict(dict):

    '''
    将dict转换为通过属性访问
    '''

    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        for key in self:
            if isinstance(self[key], dict):
                self[key] = Dict(self[key])
            elif isinstance(self[key], list):
                for i, item in enumerate(self[key]):
                    if isinstance(item, dict):
                        self[key][i] = Dict(item)

    def __getattr__(self, key):
        try:
            return self[key]
        except:
            return None
        # except KeyError as k:
            # raise AttributeError, k

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
            return True
        except:
            return False
        # except KeyError as k:
            # raise AttributeError, k

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __repr__(self):
        return '<Dict ' + dict.__repr__(self) + '>'
