#!/usr/local/bin/python2.7
# -*- coding:utf-8 -*-

import sys
import json
import Utils
import numpy as np
import pandas as pd
import xgboost
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import chardet
import jieba
import csv
from tqdm import tqdm
from nltk import word_tokenize
reload(sys)
sys.setdefaultencoding('utf-8')


class TextClassification(object):

    def __init__(self, **argv):
        print 'for now do nothing'
        print argv
        self.params = Utils.Dict(json.load(file(argv.get('config', './vfe.conf'))))
        print self.params
        print self.params['dataset']

    def trim_punct(self, content):
        punct = set(
            u''':!),.:;@~`#$%^&*()?]}¢'"、。<>〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜|｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…←↑~…*　=■● ''')
        if isinstance(content, (str, unicode)):
            return ''.join(filter(lambda x: x not in punct, content))
        elif isinstance(content, list):
            return map(lambda line: filter(lambda x: x not in punct, line), content)

    def text_process(self):

        print 'data preprocess'
        filename = self.params['dataset']
        dataset = open(filename, 'r')
        df = pd.DataFrame(columns=['classes', 'keywords'])
        print df
        count = 0
        for line in dataset.readlines():
            '''
            if count == 10000:
                break
            count += 1
            '''
            data_list = line.split('_!_')
            # print data_list
            title = unicode(data_list[3], errors='ignore')
            keywords = unicode(data_list[4].strip('\n'), errors='ignore').split(',')
            title = self.trim_punct(title)
            title_seg = set(jieba.cut_for_search(title))
            title_seg = filter(lambda x: len(x) > 1, title_seg)
            # print "keywords: ", " ".join(keywords)
            values = set(title_seg) | set(keywords)
            # print "values:", " ".join(values)
            df_tmp = pd.DataFrame([[data_list[2], " ".join(values)]], columns=['classes', 'keywords'])
            df = df.append(df_tmp, ignore_index=True, sort=False)

        # print df
        dataset.close()

        return df

    def text_classify(self):
        data = self.text_process()
        print 'clasify'
        # print data

        lbl_enc = preprocessing.LabelEncoder()
        y = lbl_enc.fit_transform(data.classes.values)

        print y[100:200]

        x_train, x_valid, y_train, y_valid = train_test_split(
            data.keywords.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)

        print x_train.shape
        print x_valid.shape
        #print x_train


        stop_word = open('./stop_word.txt', 'r')
        stop_word_list = [line.strip() for line in stop_word.readlines()]
        stop_word.close()
        tfv = NumberNormalizingVectorizer(min_df=3, max_df=0.5,
                                          max_features=None, ngram_range=(1, 2),
                                          use_idf=True, smooth_idf=True)

        # 使用TF-IDF来fit训练集和测试集（半监督学习）
        tfv.fit(list(x_train) + list(x_valid))
        xtrain_tfv = tfv.transform(x_train)
        xvalid_tfv = tfv.transform(x_valid)
        # 利用提取的TFIDF特征来fit一个简单的Logistic Regression
        clf = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial')
        clf.fit(xtrain_tfv, y_train)
        predictions = clf.predict_proba(xvalid_tfv)
        print predictions
        print y_valid
        cal_precision(predictions, y_valid)

        print ("logloss: %0.3f " % multiclass_logloss(y_valid, predictions))


def cal_precision(pred_proba, y_valid):
    pre_list = pred_proba.tolist()
    pre = [x.index(max(x)) for x in pre_list]
    print len(pre), len(y_valid)
    right = 0
    for i in range(len(pre)):
        if pre[i] == y_valid[i]:
            right += 1

    print "right: %d precision: %0.3f " % (right, float(right)/float(len(y_valid)))




def multiclass_logloss(actual, predicted, eps=1e-15):
    """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
    :param actual: 包含actual target classes的数组
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


def number_normalizer(tokens):
    """ 将所有数字标记映射为一个占位符（Placeholder）。
    对于许多实际应用场景来说，以数字开头的tokens不是很有用，
    但这样tokens的存在也有一定相关性。 通过将所有数字都表示成同一个符号，可以达到降维的目的。
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


if __name__ == '__main__':
    print 'classify data from toutiao'
    text_classification = TextClassification(config='./vfe.conf')
    text_classification.text_classify()
