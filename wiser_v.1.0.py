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


def extract_json_from_news_urls(URL_FILES, OUTPUT_FILE):
    article_list = []
    for URL_FILE in URL_FILES:
        open(DATA_PATH + URL_FILE, 'r') as inFile:
            dataset = inFile.readlines()
        for idx, url in enumerate(dataset):
            if url[0:4] == 'http':
                article_dict = {}
                art = Article(url.strip()) # don't need to strip '\n'?
                art.download()
                art.parse()
                try:
                    art.nlp()
                except newspaper.article.ArticleException:
                    print "NLP ERROR\n"
                print "{}/{} : {}".format(idx+1, 
                                          len(dataset), 
                                          art.title.encode('utf-8'))

                article_dict['url'] = art.url
                article_dict['datePublished'] = str(art.publish_date)
                article_dict['title'] = art.title
                article_dict['author'] = art.authors
                article_dict['text'] = art.text
                article_dict['summary'] = art.summary
                article_dict['keywords'] = art.keywords
                article_dict["relevant"]  = True
                article_dict["facebook"]  = True
                article_dict["likes"]  = None
                article_dict["comments"]  = None
                article_dict["category"] = None
                article_dict["tweets"]  = None
                article_dict["rt"]  = None
                article_dict["sentiment"]  = None
                article_dict["media"]  = None
                article_dict["source"]  = None
                article_dict["search_keyword"]  = None

                article_list.append(article_dict)

    with open(DATA_PATH + OUTPUT_FILE, 'w') as outFile:
        outFile.write(json.dumps(article_list, indent=4))


def predict_popularity(ARTICLE_FILE, mode='binary'):

    # get articles
    with open(DATA_PATH + ARTICLE_FILE, 'r') as inFile:
        articles_db = json.load(inFile)
    print "total number of articles: {}".format(len(articles_db))

    # get glove vectors
    glove_dict = load_glove_vectors(GLOVE_PATH, GLOVE_FILE)
    glove_len = len(glove_dict['the'])

    # make training data(vectors)
    print "extracting features..."

    NUM_SENT = 3  # first N sentences of article body to be used
    data_cnt_relev = 0
    dict_X, dict_Y = [], []
    glove_title_matrix, glove_body_matrix = np.zeros(glove_len), np.zeros(glove_len)

    for i in range(len(articles_db)):
        if SHOW_PROG_BAR: progress_bar(i,len(articles_db)-1)

        article = articles_db[i]
        if article['facebook']: # only use facebook articles
            data_cnt_relev += 1
            new_dict_X, new_dict_Y = {}, {}

            art_title = article['title'] 
            art_body  = article['text']
            art_body_head  = (' ').join(art_body.split('\n')[:NUM_SENT]) # first N sentences

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

            # money
            pattern = re.compile('sh[0-9]')   # SH is money unit in Kenya
            if pattern.search(art_title.lower()) or pattern.search(art_body_head.lower()):
                new_dict_X['money'] = True
            else:
                new_dict_X['money'] = False

            # keyword
            art_keywords = article['keywords']
            for j in range(len(art_keywords)):
                new_dict_X['keywd_{}'.format(art_keywords[j])] = True

            # media
            new_dict_X['media_{}'.format(article['media'])] = True

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

            # year & month
            new_dict_X['date_{}'.format(article.get('datePublished','')[:7])] = True

            # ---------------------
            # vector features
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

            # target : number of likes
            new_dict_Y['num_likes'] = article.get('likes',0)

            # target : number of comments
            new_dict_Y['num_comments'] = article.get('comments',0)



            # make a big list of dicts for sklearn DictVectorizer
            dict_X.append(new_dict_X)
            dict_Y.append(new_dict_Y)

    # print total time to run this part
    print ""
    print "Total Relevant   Counts : {}".format(data_cnt_relev)
    print "Total Irrelevant Counts : {}".format(len(articles_db)-data_cnt_relev)
    print "--------------------------------------"


    # vectorization
    print "Vectorizing..."
    vectorizer_X  = DictVectorizer()  # only X to be vectorized
    dictvec_trn_X = vectorizer_X.fit_transform(dict_X).toarray()

    # add word embedding as a feature
    classic_wgt = 1.0    # weights for classic features
    glove_title_wgt = 1.0    # weights for word embeddings
    glove_body_wgt = 1.0

    # concatenate word embedding to vectorized features
    glove_title_matrix = glove_title_matrix[1:]     # remove dummy first rows
    glove_body_matrix = glove_body_matrix[1:]

    dictvec_trn_X = np.hstack((dictvec_trn_X * classic_wgt,
                               glove_title_matrix * glove_title_wgt,
                               glove_body_matrix * glove_body_wgt))
    # preparing targets
    dictvec_trn_Y = []
    if  mode == 'binary':
        threshold = [10]
        for item in dict_Y:
            num_likes_comments = item['num_likes']+item['num_comments']
            if num_likes_comments > threshold[0]:
                val = 100
            else:
                val = 0
            dictvec_trn_Y.append(val)
    elif mode == 'ternary':
        threshold = [10,20]
        threshold_upper = max(threshold)
        threshold_lower = min(threshold)
        for item in dict_Y:
            num_likes_comments = item['num_likes']+item['num_comments']
            if   num_likes_comments > threshold_upper:
                val = 100
            elif num_likes_comments > threshold_lower:
                val = 50
            else:
                val = 0
            dictvec_trn_Y.append(val)
    elif mode == 'regression':
        threshold = []
        for item in dict_Y:
            num_likes_comments = item['num_likes']+item['num_comments']
            dictvec_trn_Y.append(num_likes_comments)

    # take care of NaN and INF
    dictvec_trn_X = np.nan_to_num(dictvec_trn_X)
    dictvec_trn_Y = np.nan_to_num(dictvec_trn_Y)


    #----------------
    print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
    from tsne import *
    Y = tsne(dict_vec_trn_X, 2, 50, 20.0)
    plt.scatter(Y[:,0], Y[:,1], 20, dictvec_trn_Y)
    plt.show();
    #----------------


    # choose classifier

    # Perceptron
    # clf = linear_model.Perceptron(n_jobs=-1)
    # clf = linear_model.Perceptron(penalty='l1',n_jobs=-1)
    # clf = linear_model.Perceptron(penalty='elasticnet',n_jobs=-1)

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

    # Perceptron : Passive Aggressive
    # clf = linear_model.PassiveAggressiveClassifier(n_jobs=-1)

    # Decision Tree
    #   Decision Tree needs to directly take the sparse matrix. So, use .toarray() option
    # clf = tree.DecisionTreeClassifier(criterion='gini')

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
                                       test_size=0.3,
                                       train_size=0.7,
                                       random_state=random_seed)

    # for evaluation
    acc_list, prc_list, rec_list, f1_list = [],[],[],[]
    for i, indices in enumerate(rs):
        train_index, test_index = indices[0], indices[1]
        train_X = [dictvec_trn_X[ind] for ind in train_index]
        train_Y = [dictvec_trn_Y[ind] for ind in train_index]
        test_X = [dictvec_trn_X[ind] for ind in test_index]
        test_Y = [dictvec_trn_Y[ind] for ind in test_index]

        print "---------------"
        print "iteration: {}".format(i+1)
        # train the model using the training data
        print "Training..."
        clf.fit(train_X, train_Y)

        # predict
        print "Predicting..."
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
    # URL_FILES = ['url_more.txt']
    # OUTPUT_FILE = 'articles_db_drought_the_star.json'
    # extract_json_from_news_urls(URL_FILES, OUTPUT_FILE)
    # -------------
    ARTICLE_FILE = 'articles_db_all.json'
    predict_popularity(ARTICLE_FILE, mode='binary')


if __name__ == '__main__':

    main()
