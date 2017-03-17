#!usr/bin/python
# -*- coding: utf-8 -*-

# filename: wiser_v.1.0.py 
# description: gathering news data from a list of URLs
#              and
#              making prediction of popularity on Facebook
# author: Wonchang Chung
# date: 10/25/16


import time
import json

import numpy as np

import nltk
from nltk.util import ngrams
from sklearn import cluster
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer

import gensim
from gensim import corpora
from gensim import models
from gensim import similarities
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel

import inflection
from textblob import TextBlob as tb
import newspaper
from newspaper import Article

from utils import *


DATA_PATH  = 'data/'
GLOVE_PATH = 'data/'
GLOVE_FILE = 'glove.6B.50d.txt'

SHOW_PROG_BAR = True

def predict_popularity(ARTICLE_FILE, mode='binary'):

    with open(DATA_PATH + ARTICLE_FILE, 'r') as inFile:
        articles_db = json.load(inFile)
    print "total number of articles: {}".format(len(articles_db))

    # glove vectors
    glove_dict = load_glove_vectors(GLOVE_PATH, GLOVE_FILE)
    glove_len = len(glove_dict['the'])

    print "extracting features..."
    FIRST_N = 3
    list_X, list_Y = [], []
    glove_title_matrix, glove_body_matrix = np.zeros(glove_len), np.zeros(glove_len)
    cnt_data = 0

    for article in articles_db:
        if article['facebook']: # only use facebook articles
            cnt_data += 1
            new_dict_X, new_dict_Y = {}, {}

            art_title = article['title'] 
            art_body  = article['text']
            art_body_head  = (' ').join(art_body.split('\n')[:FIRST_N]) # first N sentences

            # ---------------------
            # numerical features
            # ---------------------

            # sentiment (title)
            title = tb(art_title)
            new_dict_X['sent_title_pol'] = title.sentiment.polarity
            new_dict_X['sent_title_sbj'] = title.sentiment.subjectivity

            # sentiment (body)
            body  = tb(art_body_head)
            new_dict_X['sent_body_pol'] = body.sentiment.polarity
            new_dict_X['sent_body_sbj'] = body.sentiment.subjectivity

            # ---------------------
            # categorical features
            # ---------------------

            # keyword
            art_keywords = article['keywords']
            for j in range(len(art_keywords)):
                new_dict_X['keywd_{}'.format(art_keywords[j])] = True

            # n-grams (title)
            title_tokens = nltk.word_tokenize(art_title)
            for unigram in ngrams(title_tokens, 1):
                new_dict_X['uni_title_{}'.format(unigram)] = True
            for bigram  in ngrams(title_tokens, 2):
                new_dict_X['bi_title_{}'.format(bigram)] = True
            for trigram in ngrams(title_tokens, 3):
                new_dict_X['tri_title_{}'.format(trigram)] = True

            # n-grams (body)
            body_tokens = nltk.word_tokenize(art_body)
            for unigram in ngrams(body_tokens, 1):
                new_dict_X['uni_body_{}'.format(unigram)] = True
            for bigram  in ngrams(body_tokens, 2):
                new_dict_X['bi_body_{}'.format(bigram)] = True
            for trigram in ngrams(body_tokens, 3):
                new_dict_X['tri_body_{}'.format(trigram)] = True

            # # year & month
            # new_dict_X['date_{}'.format(article.get('datePublished','')[:7])] = True

            # # media
            # new_dict_X['media_{}'.format(article['media'])] = True


            # gazetteers

            politicians = ['Uhuru', 'Kenyatta', 'Mudavadi', 'Kamau', 'Ruto', 
               'MP', 'MCA', 'ODM', 'Jubilee', 'Boinnet', 'Muturi', 
               'Mugabe', 'Raila', 'Odinga', "Thang'wa", 'Dado', 
               'Ukur', 'Yatani', 'NYANGENA', "WANG'OMBE", 'Jama', 'Duale',]

            organizations_domestic = ['Kenya', 'KMA', 'Kenya meteorological agency', 
                                      'MET', 'KCSE', 'Government', 'Govt', 'County', 
                                      'KWS', 'KTDA', 'KCB', 'CBK', 'Parastatals',]

            organizations_international =['UN', 'United Nations', 'Red Cross', 'UNICEF', 'ICC', 'OPEC',]

            regions = ['Nairobi', 'Nakuru', 'Mombasa', 'Narok', 'Lamu', 'Garissa', 
                       'Kibwezi', 'Umani', 'Migori', 'Wajir', 'Kwale', 'Kitui', 
                       'Makueni', 'Nyanza', 'Yatta', 'Ogieks', 'Kinango', 'Mandera', 
                       'Embu', 'JKIA', 'jomo kenyatta international airport', 'Kajiado', 
                       'Twaif', 'Shamburu', 'Kinango', 'Lunga', 'Tharaka-Nithi', 'Mwea', 
                       'Dadaab', 'Ngewa', 'Tana River', 'Marsabit', 'Iten', 'Eldoret', 
                       'Pokot', 'Kilifi', 'Isiolo', 'Kiambu', 'Kisii', 'Moi', 'Kakamega',
                       'Chemelil',]

            neighbors = ['Zimbabwe', 'Djibouti', 'Somalia', 'Morocco', 'Zambia', 
                         'Ethiopia', 'Sudan', 'Uganda', 'Tanzania', 'France', 'Paris', 
                         'Rwanda', 'Burundi', 'Australia', 'Israel', 'Europe', 'EU', 'America', 'World',]

            climates = ['Climate', 'Weather', 'Paris',]

            agricultures = ['Agriculture', 'Farmer', 'Maize', 'Tea', 'Potato', 'Camel', 
                            'Plant', 'livestock', 'cassava', 'capsicum', 'rice',]

            business = ['price', 'stock', 'insurance', 'insurer',]

            monetary = ['money', 'budget', 'cash', 'share', 'inflation', 'deflation', 'price',]

            pope = ['pope', 'francisco',]

            # money
            new_dict_X['money'] = False
            pattern = re.compile('sh[0-9]')   # SH is money unit in Kenya
            if (pattern.search(art_title.lower())):
                new_dict_X['money'] = True
            for word in monetary:
                if word.lower() in art_title:
                    new_dict_X['money'] = True

            # politicians
            new_dict_X['politician'] = False
            for word in politicians:
                if word.lower() in art_title:
                    new_dict_X['politician'] = True

            # domestic organizations
            new_dict_X['organ_dom'] = False
            for word in organizations_domestic:
                if word.lower() in art_title:
                    new_dict_X['organ_dom'] = True

            # international organizations
            new_dict_X['organ_int'] = False
            for word in organizations_international:
                if word.lower() in art_title:
                    new_dict_X['organ_int'] = True

            # regions
            new_dict_X['region'] = False
            for word in regions:
                if word.lower() in art_title:
                    new_dict_X['region'] = True

            # neighbors
            new_dict_X['neighbor'] = False
            for word in neighbors:
                if word.lower() in art_title:
                    new_dict_X['neighbor'] = True

            # climate
            new_dict_X['climate'] = False
            for word in climates:
                if word.lower() in art_title:
                    new_dict_X['climate'] = True

            # agriculture
            new_dict_X['agri'] = False
            for word in agricultures:
                if word.lower() in art_title:
                    new_dict_X['agri'] = True

            # business
            new_dict_X['biz'] = False
            for word in business:
                if word.lower() in art_title:
                    new_dict_X['biz'] = True

            # pope
            new_dict_X['pope'] = False
            for word in pope:
                if word.lower() in art_title:
                    new_dict_X['pope'] = True

            # ---------------------
            # word vector features
            # ---------------------

            # average glove vector (title)
            glove_avg_title = word_emb_avg(art_title, glove_dict)
            glove_title_matrix = np.vstack((glove_title_matrix, glove_avg_title))

            # average glove vector (body)
            glove_avg_text = word_emb_avg(art_body_head, glove_dict)            
            glove_body_matrix = np.vstack((glove_body_matrix, glove_avg_text))

            # ---------------------
            # targets
            # ---------------------

            new_dict_Y['num_likes'] = article.get('likes',0)
            new_dict_Y['num_comments'] = article.get('comments',0)

            # make a big list of dicts for sklearn DictVectorizer
            list_X.append(new_dict_X)
            list_Y.append(new_dict_Y)

    print "Total Relevant Counts : {}".format(cnt_data)
    print "--------------------------------------"


    # vectorization
    print "Vectorizing..."
    vectorizer_X = DictVectorizer()  # only X to be vectorized
    dictvec_trn_X = vectorizer_X.fit_transform(list_X).toarray()

    # add word embedding as a feature
    classic_wgt = 1.0    # weights for classic features
    glove_title_wgt = 1.0    # weights for word embeddings
    glove_body_wgt = 1.0

    glove_title_matrix = glove_title_matrix[1:]     # remove dummy first rows
    glove_body_matrix = glove_body_matrix[1:]

    dictvec_trn_X = np.hstack((dictvec_trn_X * classic_wgt,
                               glove_title_matrix * glove_title_wgt,
                               glove_body_matrix * glove_body_wgt))
    # preparing targets
    threshold = 20
    dictvec_trn_Y = []
    if  mode == 'binary':
        for item in list_Y:
            num_likes_comments = item['num_likes'] + item['num_comments']
            if num_likes_comments > threshold:
                val = 100
            else:
                val = 0
            dictvec_trn_Y.append(val)
    elif mode == 'regression':
        for item in list_Y:
            num_likes_comments = item['num_likes'] + item['num_comments']
            dictvec_trn_Y.append(num_likes_comments)

    # take care of NaN and INF
    dictvec_trn_X = np.nan_to_num(dictvec_trn_X)
    dictvec_trn_Y = np.nan_to_num(dictvec_trn_Y)


    ####################
    # choose classifier

    # Perceptron
    # clf = linear_model.Perceptron(n_jobs=-1)
    # clf = linear_model.Perceptron(penalty='l1',n_jobs=-1)
    # clf = linear_model.Perceptron(penalty='elasticnet',n_jobs=-1)
    clf = linear_model.PassiveAggressiveClassifier(n_jobs=-1)

    # SVC
    # clf = svm.SVC(kernel='linear')
    # clf = svm.SVC(kernel='poly', probability=True, degree=3)
    # clf = svm.SVC(kernel='rbf', probability=True)
    # clf = svm.SVC(kernel='sigmoid')

    # SVR
    # clf = svm.SVR(kernel='linear')

    # Logistic Regression
    # clf = linear_model.LogisticRegression(penalty='l2', C=1)

    # Linear Regression    
    # clf = linear_model.LinearRegression()

    # Decision Tree
    #   Decision Tree needs to directly take the sparse matrix. So, use .toarray() option
    # clf = tree.DecisionTreeClassifier(criterion='gini')
    # clf = tree.DecisionTreeClassifier(criterion='entropy')

    # Random Forest
    # clf = ensemble.RandomForestClassifier(max_depth=3)
    # clf = ensemble.RandomForestRegressor(max_depth=4)

    # AdaBoost
    # clf = ensemble.AdaBoostClassifier()

    # KNN
    # clf = neighbors.KNeighborsClassifier(n_neighbors=2)

    # Naive Bayes
    # clf = naive_bayes.GaussianNB()
    # clf = naive_bayes.MultinomialNB()
    # clf = naive_bayes.BernoulliNB()


    # cross validation
    random_seed = 1
    rs = cross_validation.ShuffleSplit(n=len(dictvec_trn_X),
                                       n_iter=10,
                                       test_size=0.1,
                                       train_size=0.9,
                                       random_state=random_seed)

    # for evaluation
    acc_list, prc_list, rec_list, f1_list = [], [], [], []
    for i, indices in enumerate(rs):
        train_index, test_index = indices[0], indices[1]
        train_X = [dictvec_trn_X[ind] for ind in train_index]
        train_Y = [dictvec_trn_Y[ind] for ind in train_index]
        test_X = [dictvec_trn_X[ind] for ind in test_index]
        test_Y = [dictvec_trn_Y[ind] for ind in test_index]

        print "---------------"
        print "iteration: {}".format(i+1)

        # train and predict
        clf.fit(train_X, train_Y)
        result = clf.predict(test_X)

        # evaluate
        acc, prc, rec, f1 = evaluate(result, 
                                     test_Y, 
                                     threshold=threshold, 
                                     mode=mode, 
                                     verbose=True)
        acc_list.append(acc)
        prc_list.append(prc)
        rec_list.append(rec)
        f1_list.append(f1)

    print "----------------------------------------------"
    print "Average Accuracy  : {0:.2f}".format(np.mean(acc_list))
    if mode == 'binary':
        print "Average Precision : {0:.2f}".format(np.mean(prc_list))
        print "Average Recall    : {0:.2f}".format(np.mean(rec_list))
        print "Average F1        : {0:.2f}".format(np.mean(f1_list ))


def main():
    ARTICLE_FILE = 'articles_db_all.json'
    predict_popularity(ARTICLE_FILE, mode='binary')


if __name__ == '__main__':

    main()
