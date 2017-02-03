#!usr/bin/python
# -*- coding: utf-8 -*-

# filename: wiser_v.1.0.py 
# description: gathering news data from a list of URLs
#              and
#              making prediction of popularity on Facebook
# author: Wonchang Chung
# date: 10/25/16

from utils import *

import numpy as np
import time
import json
import nltk
import inflection
from textblob import TextBlob as tb
import newspaper
from newspaper import Article
from sklearn import svm, linear_model, cluster, neighbors, tree, ensemble, naive_bayes
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer

DATA_PATH = 'data/'

# GloVe word embedding
WDEMB_PATH = 'data/'
WDEMB_FILE = 'glove.6B.50d.txt'

SHOW_PROG_BAR = True

def extract_json_from_news_urls(URL_FILES, OUTPUT_FILE):
    '''
    extract JSON from news articles
    '''

    article_list = []
    for URL_FILE in URL_FILES:
        # load news article URLs
        open(DATA_PATH + URL_FILE, 'r') as inFile:
            dataset = inFile.readlines()
        # read each URL
        for idx, url in enumerate(dataset):
            if url[0:4] == 'http':
                article_dict = {}
                art = Article(url.strip())          # don't need to strip '\n'?
                art.download()
                art.parse()
                try:
                    art.nlp()
                except newspaper.article.ArticleException:
                    print "NLP ERROR\n"
                print "{}/{} : {}".format(idx+1,len(dataset),art.title.encode('utf-8'))

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

    # write to output file
    with open(DATA_PATH + OUTPUT_FILE, 'w') as outFile:
        outFile.write(json.dumps(article_list, indent=4))


def predict_popularity(ARTICLE_FILE, mode='binary'):
    '''
    load JSON data
    and
    predict popularity of each news articles in Facebook
    '''
    
    DATA_PATH = 'data/'

    # load raw data in JSON format
    print "loading articles db file..."
    with open(DATA_PATH + ARTICLE_FILE, 'r') as inFile:
        articles_db = json.load(inFile)
    print "  total number of articles: {}".format(len(articles_db))

    # load word embedding
    print "loading Word Embedding..."
    with open(WDEMB_PATH + WDEMB_FILE, 'r') as infile:
        wdembset = infile.readlines()
    itemized_wdemb = [item.split() for item in wdembset]
    wdemb_dict = {}
    for idx, vec in enumerate(itemized_wdemb):
        if SHOW_PROG_BAR: progress_bar(idx,len(itemized_wdemb))
        wdemb_dict[vec[0]] = np.asarray(vec[1:], dtype=np.float32) # cast as float 32, which is the format of GloVe
    print ""


    # make training data(vectors)
    print "extracting features..."

    # first N sentences of article body to be used
    NUM_SENT = 3

    data_cnt_relev  = 0
    dict_X, dict_Y  = [], []
    len_wdemb = len(wdemb_dict['the'])
    wdemb_title_matrix, wdemb_text_matrix = np.zeros(len_wdemb), np.zeros(len_wdemb)
    for i in range(len(articles_db)):
        if SHOW_PROG_BAR: progress_bar(i,len(articles_db)-1)
        if articles_db[i]['facebook']:

            data_cnt_relev += 1
            new_dict_X, new_dict_Y = {}, {}

            art_title = articles_db[i]['title'] 
            art_body  = articles_db[i]['text']
            art_body_head  = (' ').join(art_body.split('\n')[:NUM_SENT]) # first N sentences

            # feature : money
            pattern = re.compile('sh[0-9]')   # SH is money unit in Kenya
            if pattern.search(art_title.lower()) or pattern.search(art_body_head.lower()):
                new_dict_X['money'] = True
            else:
                new_dict_X['money'] = False

            # feature : keyword
            for j in range(10):
                try:
                    new_dict_X['keywd_{}'.format(j+1)] = articles_db[i]['keywords'][j]
                except:
                    new_dict_X['keywd_{}'.format(j+1)] = ''

            # feature : sentiment (title)
            title = tb(art_title)
            new_dict_X['sent_title_pol'] = title.sentiment.polarity
            new_dict_X['sent_title_sbj'] = title.sentiment.subjectivity

            # feature : sentiment (body)
            body = tb(art_body_head)
            new_dict_X['sent_body_pol'] = body.sentiment.polarity
            new_dict_X['sent_body_sbj'] = body.sentiment.subjectivity

            # feature : media
            new_dict_X['media'] = articles_db[i]['media']

            # feature : n-grams (title)
            title_uni = nltk.word_tokenize(art_title)
            for unigram in title_uni:
                new_dict_X['uni_title_' + unigram] = True
            title_bi  = [(title_uni[j], title_uni[j+1]) for j in range(len(title_uni)-1)]
            for bigram in title_bi:
                new_dict_X['bi_title_' + bigram[0] + '_' + bigram[1]] = True
            title_tri = [(title_uni[j], title_uni[j+1], title_uni[j+2]) for j in range(len(title_uni)-2)]
            for trigram in title_tri:
                new_dict_X['tri_title_' + trigram[0] + '_' + trigram[1] + '_' + trigram[2]] = True

            # feature : n-grams (body)
            text_uni = nltk.word_tokenize(art_body)
            for unigram in text_uni:
                new_dict_X['uni_text_' + unigram] = True
            text_bi  = [(text_uni[j], text_uni[j+1]) for j in range(len(text_uni)-1)]
            for bigram in text_bi:
                new_dict_X['bi_text_' + bigram[0] + '_' + bigram[1]] = True
            text_tri = [(text_uni[j], text_uni[j+1], text_uni[j+2]) for j in range(len(text_uni)-2)]
            for trigram in text_tri:
                new_dict_X['tri_text_' + trigram[0] + '_' + trigram[1] + '_' + trigram[2]] = True

            # feature : year & month
            new_dict_X['date'] = articles_db[i].get('datePublished','')[:7]
            
            # target : number of likes
            new_dict_Y['num_likes'] = articles_db[i].get('likes',0)

            # target : number of comments
            new_dict_Y['num_comments'] = articles_db[i].get('comments',0)

            # make a big list of dicts for sklearn DictVectorizer
            dict_X.append(new_dict_X)
            dict_Y.append(new_dict_Y)

            # feature : word embedding (title)
            wdemb_avg_title = word_emb_avg(art_title, wdemb_dict)
            wdemb_title_matrix = np.vstack((wdemb_title_matrix, wdemb_avg_title))

            # feature : word embedding (body)
            wdemb_avg_text = word_emb_avg(art_body_head, wdemb_dict)            
            wdemb_text_matrix = np.vstack((wdemb_text_matrix, wdemb_avg_text))

    wdemb_title_matrix = wdemb_title_matrix[1:]     # remove dummy first rows
    wdemb_text_matrix  = wdemb_text_matrix[1:]

    # print total time to run this part
    print ""
    print "Total Data Count    : {}".format(data_cnt_relev)
    print "Total Irrelevants   : {}".format(len(articles_db)-data_cnt_relev)
    print "--------------------------------------"


    # vectorization
    print "Vectorizing..."
    vectorizer_X  = DictVectorizer()  # only X to be vectorized
    dictvec_trn_X = vectorizer_X.fit_transform(dict_X).toarray()

    # add word embedding as a feature
    classic_wgt     = 1.0    # weights for classic features
    wdemb_title_wgt = 1.0    # weights for word embeddings
    wdemb_text_wgt  = 1.0

    # concatenate word embedding to vectorized features
    dictvec_trn_X = np.hstack((dictvec_trn_X * classic_wgt,
                               wdemb_title_matrix * wdemb_title_wgt,
                               wdemb_text_matrix * wdemb_text_wgt))
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
        test_X  = [dictvec_trn_X[ind] for ind in test_index]
        test_Y  = [dictvec_trn_Y[ind] for ind in test_index]

        print "---------------"
        print "iteration: {}".format(i+1)
        # train the model using the training data
        print "Training..."
        clf.fit(train_X, train_Y)

        # predict
        print "Predicting..."
        result = clf.predict(test_X)

        # evaluate
        acc, prc, rec, f1 = evaluate(result, test_Y, threshold=threshold , mode=mode, verbose=True)
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


if __name__ == '__main__':

    # URL_FILES = ['url_more.txt']
    # OUTPUT_FILE = 'articles_db_drought_the_star.json'

    # extract_json_from_news_urls(URL_FILES, OUTPUT_FILE)

    # -------------

    ARTICLE_FILE = 'articles_db_all.json'

    predict_popularity(ARTICLE_FILE, mode='binary')


