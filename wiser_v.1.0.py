#!usr/bin/python
# -*- coding: utf-8 -*-

# filename: wiser_v.1.0.py 
# description: gathering news data from a list of URLs
#              and
#              making prediction of popularity on Facebook
# author: Wonchang Chung
# date: 10/25/16

from utils import *

import time
import json

import numpy as np

import newspaper
from newspaper import Article

import nltk
from textblob import TextBlob as tb
import inflection

# scikit-learn family
from sklearn import svm, linear_model, cluster, neighbors, tree, ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn import cross_validation


DATA_PATH = 'data/'
# URL_FILE = 'url.txt'
URL_FILES = ['url_climate.txt','url_drought.txt','url_weather.txt']
# OUTPUT_FILE = 'articles_db.json'
OUTPUT_FILE = 'articles_db_new.json'

# GloVe word embedding
WDEMB_PATH = 'data/'
WDEMB_FILE = 'glove.6B.50d.txt'


def extract_json_from_news_urls():
    '''
    extract json from news articles
    '''

    # initiate a list for making .json
    article_list = []

    for URL_FILE in URL_FILES:

        # load news article url file
        inFile = open(DATA_PATH + URL_FILE, 'r')
        dataset = inFile.readlines()
        inFile.close()

        i = 0

        for url in dataset:
            i += 1
            if url[0:4] == 'http':
                
                article_dict = {}

                print "---------------------------------------------"
                art = Article(url.strip())          # don't need to strip '\n'?
                art.download()
                art.parse()
                try:
                    art.nlp()
                except newspaper.article.ArticleException:
                    print "NLP ERROR\n"
                print str(i)+"/"+str(len(dataset))+" : "+art.title + "\r"

                article_dict['url'] = art.url
                article_dict['media'] = None
                article_dict['datePublished'] = str(art.publish_date)
                article_dict['title'] = art.title
                article_dict['author'] = art.authors
                article_dict['text'] = art.text

                article_dict['summary'] = art.summary
                article_dict['keywords'] = art.keywords

                article_dict["category"] = None
                article_dict["rt"]  = None
                article_dict["relevant"]  = None
                article_dict["sentiment"]  = None
                article_dict["media"]  = None
                article_dict["comments"]  = None
                article_dict["datePublished"]  = None
                article_dict["source"]  = None
                article_dict["facebook"]  = None
                article_dict["likes"]  = None
                article_dict["tweets"]  = None
                article_dict["search_keyword"]  = URL_FILE[4:-4]

                # print article_dict

                article_list.append(article_dict)

        # print article_list
        print "---------------------------------------------"

    # print json.dumps(article_list)

    # print article_list
    outFile = open(DATA_PATH + OUTPUT_FILE, 'w')
    outFile.write(json.dumps(article_list))
    outFile.close()


def predict():

    # load raw data in JSON format
    DATA_PATH = 'data/'
    # ARTICLE_FILE = 'articles_db.json'
    # ARTICLE_FILE = 'articles_db_new.json'
    ARTICLE_FILE = 'articles_db_temp.json'

    print "loading articles db file..."
    inFile = open(DATA_PATH + ARTICLE_FILE, 'r')
    articles_db = json.load(inFile)
    inFile.close()
    print "loading articles db - done."
    print "  total number of articles: " + str(len(articles_db))


    # load word embedding

    print "loading Word Embedding..."
    infile = open(WDEMB_PATH + WDEMB_FILE, 'r')
    wdembset = infile.readlines()
    infile.close()

    # itemizing every lines
    itemized_wdemb = [item.split() for item in wdembset]

    # making dict of word embedding
    wdemb_dict = {}

    for vec in itemized_wdemb:
        '''
        NEED TO CAST as float?
        '''
        wdemb_dict[vec[0]] = np.asarray(vec[1:])

    print "done."


    # make training data(vectors)

    print "making the dictionary..."
    t_de = time.clock()

    # get the maximum number of each features
    cat_max     = 0
    source_max  = 0
    keywords_max= 0
    for i in range(len(articles_db)):
        try:
            if len(articles_db[i]['category']) > cat_max:
                cat_max    = len(articles_db[i]['category'])
            if len(articles_db[i]['source']) > source_max:
                source_max = len(articles_db[i]['source'])
            if len(articles_db[i]['keywords']) > keywords_max:
                keywords_max = len(articles_db[i]['keywords'])
        except:
            pass

    # for counting the number of real data
    data_count  = 0

    # data for DictVectorizer()
    dict_X  = []
    dict_Y  = []

    # counter for irrelevant articles
    not_relevant_cnt = 0 

    # initialize a dummy vector with word embedding matrix
    wdemb_title_matrix    = np.zeros(50)
    wdemb_text_matrix     = np.zeros(50)

    for i in range(len(articles_db)):
        if articles_db[i]['relevant'] and ( articles_db[i]['comments'] > 0 or articles_db[i]['likes'] > 0):

            # temporary dicts of input and output
            new_dict_X = {}
            new_dict_Y = {}

            # feature : category
            for j in range(cat_max):
                try:
                    new_dict_X['cat_' + str(j+1)] = articles_db[i]['category'][j]
                except:
                    new_dict_X['cat_' + str(j+1)] = ''

            # feature : source
            list_organization = [['UN','U.N.','United Nations','United Nation'],['UNOCHA','OCHA','United Nations Office of Coordination and Humanitarian Affairs','United Nations Office for the Coordination of Humanitarian Affairs','Office for the Coordination of Humanitarian Affairs'],["Central Bank of Kenya’s",'Central Bank of Kenya','CBK'],['Kenya Tea Development Agency','KTDA'],['Nairobi Coffee Exchange','NCE'],['International Coffee Organisation','ICO'],['Monetary Policy Committee','MPC'],['Kenya Farmers Association','KFA'],['Pan Africanist Congress','PAC'],['International Water Management Institute','IWMI'],['Cereal Growers Association','CGA'],['Kenya Flower Council','KFC'],['Oserian Development Company','ODC'],['Kenya National Bureau of Statistics','KNBS'],['National Oceanic and Atmospheric Administration','NOAA'],['National Aeronautics and Space Administration','NASA'],['Agriculture, Fisheries and Food Authority','AFFA'],['Inter-Governmental Authority on Development','IGAD'],['National Cereals and Produce Board','NCPB'],['EU','European Union'],["United Nations Children’s Fund",'UNICEF'],['World Health Organisation','WHO'],['Coalition for Reforms and Democracy','cord'],['Sustainable Development Solutions Network','SDSN'],['African Journal of Food, Agriculture, Nutrition and Development','AJFAND'],['National Transport and Safety Authority','NTSA'],['National Environmental Management Authority','nema'],['World Food Programme','World Food Program','WFP'],['Kenya Veterinary Vaccines Institute','KEVEVAPI'],['Kenya Red Cross','KRC'],['Kenya Meteorological Service','Kenya Meteorological Department','Meteorological Services','Met services','Meteorological Service','KMS'],['Agriculture and Food Security','CCAFS'],['Regional Agricultural Trade Intelligence Network','RATIN'],['Orange Democratic Movement','ODM'],['New Kenya Cooperative Creameries','Kenya Cooperative Creameries','KCC'],['Kenya Airports Authority','KAA']]

            for j in range(source_max):
                try:
                    src = articles_db[i]['source'][j]
                    src = src.lower()

                    # Organization Normalization
                    for organiz in list_organization:
                        organiz = [inflection.singularize(org).lower() for org in organiz]
                        # The를 빼는 것도 넣으면 정확도가 조금은 올라갈 것이다

                        for k in range(len(organiz)):
                            if src == organiz[k]:
                                src = organiz[len(organiz)-1]
                                # src = organiz[len(organiz)]

                    new_dict_X['source_' + str(j+1)] = src
                except:
                    new_dict_X['source_' + str(j+1)] = ''

            # feature : keyword
            for j in range(keywords_max):
                try:
                    new_dict_X['keyword_' + str(j+1)] = articles_db[i]['keywords'][j]
                except:
                    new_dict_X['keyword_' + str(j+1)] = ''

            # # feature : sentiment (annotated)
            # new_dict_X['sentiment_body_anno'] = articles_db[i]['sentiment'][0]


            # feature : sentiment (body)
            body = tb(articles_db[i]['text'])

            # #     polarity     of the full body
            # new_dict_X['sentiment_body_full_pol'] = body.sentiment.polarity
            # #     subjectivity of the full body
            # new_dict_X['sentiment_body_full_sbj'] = body.sentiment.subjectivity
            # #     averaged polarity of all sentences in the full body
            # ###### THIS ONE GIVES ME ERROR
            # new_dict_X['sentiment_body_avg_pol']  = float(np.mean([sentence.sentiment.polarity for sentence in body.sentences]))
            # #     averaged subjectivity of all sentences in the full body
            # ###### THIS ONE GIVES ME ERROR
            # new_dict_X['sentiment_body_avg_sbj']  = float(np.mean([sentence.sentiment.subjectivity for sentence in body.sentences]))
            # # 그건 그런데 센티멘트의 수치 크기를 어떻게 반영할 것인가?


            # feature : sentiment (title)
            title = tb(articles_db[i]['title'])
            new_dict_X['sentiment_title_pol'] = title.sentiment.polarity
            new_dict_X['sentiment_title_sbj'] = title.sentiment.subjectivity


            # # feature : media
            # new_dict_X['media'] = articles_db[i]['media'][0]


            # feature : n-grams (body)
            # 원래 바이그램과 트라이그램은 시작과 끝을 잘 챙겨줘야 하지만 일단은 무시했다.
            text_uni = nltk.word_tokenize(articles_db[i]['text'])
            for unigram in text_uni:
                new_dict_X['uni_text_' + unigram] = True

            text_bi  = [(text_uni[j], text_uni[j+1]) for j in range(len(text_uni)-1)]
            for bigram in text_bi:
                new_dict_X['bi_text_' + bigram[0] + '_' + bigram[1]] = True

            text_tri = [(text_uni[j], text_uni[j+1], text_uni[j+2]) for j in range(len(text_uni)-2)]
            for trigram in text_tri:
                new_dict_X['tri_text_' + trigram[0] + '_' + trigram[1] + '_' + trigram[2]] = True


            # feature : n-grams (title)
            # 원래 바이그램과 트라이그램은 시작과 끝을 잘 챙겨줘야 하지만 일단은 무시했다.
            title_uni = nltk.word_tokenize(articles_db[i]['title'])
            for unigram in title_uni:
                new_dict_X['uni_title_' + unigram] = True

            title_bi  = [(title_uni[j], title_uni[j+1]) for j in range(len(title_uni)-1)]
            for bigram in title_bi:
                new_dict_X['bi_title_' + bigram[0] + '_' + bigram[1]] = True

            title_tri = [(title_uni[j], title_uni[j+1], title_uni[j+2]) for j in range(len(title_uni)-2)]
            for trigram in title_tri:
                new_dict_X['tri_title_' + trigram[0] + '_' + trigram[1] + '_' + trigram[2]] = True



            # feature : date
            # 월 정보만 넣자

            # new_dict_X['date'] = articles_db[i]['datePublished'][:7]

            
            # # target : number of tweets
            # new_dict_Y['num_tweets'] = articles_db[i]['tweets']

            # # target : number of RT(retweets)
            # new_dict_Y['num_RTs'] = articles_db[i]['rt']

            # target : number of likes
            new_dict_Y['num_likes'] = articles_db[i]['likes']

            # target : number of comments
            new_dict_Y['num_comments'] = articles_db[i]['comments']


            # check whether this article was a real data
            if new_dict_X == {}:
                pass
            else:
                data_count += 1

            # make a big list of dicts for DictVectorizer()
            dict_X.append(new_dict_X)
            dict_Y.append(new_dict_Y)


            # feature : WORD EMBEDDING
            # of title
            wdemb_avg_title = word_emb_avg(articles_db[i]['title'], wdemb_dict)
            try:
                wdemb_title_matrix = np.vstack((wdemb_title_matrix, wdemb_avg_title))
            except:  # zero vector case
                wdemb_title_matrix = np.vstack((wdemb_title_matrix, np.zeros(50)))
            print "word embedding of title added."

            # of text body
            NUM_SENT = 3
            wdemb_avg_text = word_emb_avg((' ').join(articles_db[i]['text'].split('\n')[:NUM_SENT]), wdemb_dict)            
            try:
                wdemb_text_matrix = np.vstack((wdemb_text_matrix, wdemb_avg_text))
            except:  # zero vector case
                wdemb_text_matrix = np.vstack((wdemb_text_matrix, np.zeros(50)))
            print "word embedding of textbody added."

        else:
            # count articles which are NOT relevant
            not_relevant_cnt += 1

    wdemb_title_matrix    = wdemb_title_matrix[1:]     # removing the dummy first row
    wdemb_text_matrix     = wdemb_text_matrix[1:]      # removing the dummy first row

    print str(not_relevant_cnt) + " articles were NOT relevant"

    # print total time to run this part
    print "Total Data Count    : " + str(data_count)
    print "Data Extraction Time: " + str(time.clock() - t_de) + ' sec'
    print "--------------------------------------"


    # vectorization

    t_de = time.clock()
    print "Vectorizing..."

    #  load vectorizer
    vectorizer_X = DictVectorizer()
    vectorizer_Y = DictVectorizer()   # TODO : 결국 Υ 는 여기서 벡터라이징 안혔음

    # dictvec_trn_X = vectorizer_X.fit_transform(dict_X)
    dictvec_trn_X = vectorizer_X.fit_transform(dict_X).toarray()      # if the matrix is not that big

    # add word embedding as a feature

    # weight for word embedding
    classic_wgt        = 1.0
    wdemb_title_wgt    = 1.0
    wdemb_text_wgt     = 1.0

    # appending word embedding to vectorized features
    dictvec_trn_X = np.hstack((dictvec_trn_X * classic_wgt, wdemb_title_matrix * wdemb_title_wgt, wdemb_text_matrix * wdemb_text_wgt))
    dictvec_trn_Y = []

    # preparing targets
    for item in dict_Y:
        # dictvec_trn_Y.append(item['num_likes'])
        # dictvec_trn_Y.append(item['num_comments'])
        if max(item['num_likes'],-float('Inf'))+max(item['num_comments'],-float('Inf')) > 10:
        # if item['num_likes'] > 10:
            val = 20
        else:
            val = 0
        dictvec_trn_Y.append(val)
        # dictvec_trn_Y.append(max(item['num_likes'],-float('Inf'))+max(item['num_comments'],-float('Inf')))
        

    # take care of NaN and INF
    dictvec_trn_X = np.nan_to_num(dictvec_trn_X)
    dictvec_trn_Y = np.nan_to_num(dictvec_trn_Y)

    print "DictVectorizng Time: " + str(time.clock() - t_de) + ' sec'
    print "--------------------------------------"



    # choose classifier

    # Decision Tree
    #   Decision Tree needs to directly take the sparse matrix. So, use .toarray() option
    # clf = tree.DecisionTreeClassifier(criterion='gini')

    # Perceptron : Passive Aggressive
    # clf = PassiveAggressiveClassifier(n_jobs=-1)

    # Perceptron
    # clf = Perceptron(n_jobs=-1)
    # clf = Perceptron(penalty='l1',n_jobs=-1)
    # clf = Perceptron(penalty='elasticnet',n_jobs=-1)
    # clf = Perceptron(n_jobs=-1)

    # Linear Regression    
    # clf = linear_model.LinearRegression()

    # SGD
    # clf = SGDClassifier()

    # KNN
    #   이거는 거의 값을 0으로 만들어 버린다 안될 듯
    # clf = neighbors.KNeighborsClassifier()

    # SVM
    # clf = svm.SVC(kernel='linear')                                # 이거 다메
    # clf = svm.SVC(kernel='poly', degree=3, gamma=1/X_dim*10)      # 이건 동작 안함
    # clf = svm.SVC(kernel='poly', probability=True, degree=3)      # 이것도 다메 SVM은 전반적으로 그냥 한 값으로 만들어버리네?
    # clf = svm.SVC(kernel='rbf', gamma=1/X_dim*10)                 # 이건 동작 안함
    # clf = svm.SVC(kernel='rbf', probability=True)                 # 이거 다메
    # clf = svm.SVC(kernel='sigmoid')                               # 이거 다메
    # clf = svm.SVC(kernel='precomputed')                           # 이거 동작 안함

    # Logistic Regression
    clf = linear_model.LogisticRegression(penalty='l2', C=1)



    # cross validation

    random_seed = 1

    # normal ShuffleSplit
    rs = cross_validation.ShuffleSplit(n=len(dictvec_trn_X), n_iter=10, test_size=0.3, train_size=0.7, random_state=random_seed)
    # StratifiedShuffleSplit : train set 과 test set 을 나누는데 있어서, Y 의 비율을 일정하게 나누는 것인데, 이 경우 하나의 Y 값들이 여러개가 있어야 하겠지. 지금처럼은 안된다.
    # rs = cross_validation.StratifiedShuffleSplit(dict_Y, n_iter=1, test_size=0.2, train_size=0.8, random_state=random_seed)

    acc_list, prc_list, rec_list, f1_list = [],[],[],[]
    hot_set, not_set = set(), set()
    hot_vecs_list, not_vecs_list = [],[]
    for train_index, test_index in rs:

        train_X = [dictvec_trn_X[ind] for ind in train_index]
        train_Y = [dictvec_trn_Y[ind] for ind in train_index]

        test_X  = [dictvec_trn_X[ind] for ind in test_index]
        test_Y  = [dictvec_trn_Y[ind] for ind in test_index]


        # start timer
        t_tc = time.clock()
        print "Training..."

        # train the model using the training data
        clf.fit(train_X, train_Y)


        # print total time to run this part
        print "Training Time: " + str(time.clock() - t_tc) + ' sec'
        print "--------------------------------------"


        # predict

        print "Predicting..."
        result = clf.predict(test_X)


        # evaluate

        acc, prc, rec, f1 = evaluate(result, test_Y)
        acc_list.append(acc)
        prc_list.append(prc)
        rec_list.append(rec)
        f1_list.append(f1)

        # print out HOTs and NOTs
        #   HOTs
        print "----------------------------------------------"
        print "HOT articles"
        print "----------------------------------------------"
        for i, res in enumerate(result):
            if res > 10:
                hot_set.add(articles_db[test_index[i]]['title'])
                hot_vecs_list.append(wdemb_title_matrix[i])


        #   NOTs
        print "----------------------------------------------"
        print "NOT articles"
        print "----------------------------------------------"
        for i, res in enumerate(result):
            if res <= 10:
                not_set.add(articles_db[test_index[i]]['title'])
                not_vecs_list.append(wdemb_title_matrix[i])



    print "----------------------------------------------"
    print "Average Accuracy  : " + str(np.mean(acc_list))
    print "Average Precision : " + str(np.mean(prc_list))
    print "Average Recall    : " + str(np.mean(rec_list))
    print "Average F1        : " + str(np.mean(f1_list))




if __name__ == '__main__':

    # extract_json_from_news_urls()

    predict()