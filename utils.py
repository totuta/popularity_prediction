#!usr/bin/python
# -*- coding: utf-8 -*-

# filename: utils.py 
# description: util functions for WISER project
# author: Wonchang Chung
# date: 12/19/16

from __future__ import division

import sys
import time
import re
import signal
import json

import numpy as np
from scipy.stats.stats import pearsonr

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from sklearn.metrics.pairwise import cosine_similarity

import inflection
import urlunshort
import urlparse

import newspaper
from newspaper import Article


def progress_bar(current_step, total_step, graph_step=2.5):
    '''
    displaying progress bar

    current_step : current step during the process
                   should start from 1
                   should be given
    total_step : total number of steps to finish the process
    graph_step : one unit(in percentage) in the progress bar
    '''

    percent = round(current_step/total_step*100,1)
    percent_bar = int(percent/graph_step)

    # display
    sys.stdout.write('\r')
    sys.stdout.write('[')
    for i in range(percent_bar): sys.stdout.write('=')
    if percent < 100: sys.stdout.write('>')
    for i in range(int(100/graph_step-1-percent_bar)): sys.stdout.write('.')
    sys.stdout.write(']')
    sys.stdout.write('   '+str(current_step)+'/'+str(total_step)+'    '+str(percent)+'%')
    sys.stdout.flush() # important


class TimeoutException(Exception):   # Custom exception class
    """
    to time-out a certain function
    """
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)


def extract_json_from_news_urls():

    article_list = []
    for URL_FILE in URL_FILES:
        with open(DATA_PATH + URL_FILE, 'r') as inFile: # load news article url file
            dataset = inFile.readlines()
        for idx, url in enumerate(dataset):
            if url[0:4] == 'http':
                article_dict = {}
                print "{}/{} : {}\r".format(idx+1,len(dataset),art.title)
                art = Article(url.strip())
                art.download()
                art.parse()
                try:
                    art.nlp()
                except newspaper.article.ArticleException:
                    print "NLP ERROR\n"

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

                article_list.append(article_dict)

    with open(DATA_PATH + OUTPUT_FILE, 'w') as outFile:
        outFile.write(json.dumps(article_list, indent=4))


def extract_json():
    '''
    extract a compact JSON file from the raw json of downloaded tweets
    '''

    DATA_PATH = ''
    TWEET_FILE = 'tweets_elnino_0125-0501.json'
    # TWEET_FILE = 'tweets_lanina_0211-0429.json'
    OUT_FILE = 'tweets_elnino_0125-0501_simple.json'
    # OUT_FILE = 'tweets_lanina_0211-0429_simple.json'

    inFile = open(DATA_PATH + TWEET_FILE, 'r')
    # dataset = inFile.readlines()
    temp_json = json.load(inFile)
    inFile.close()

    KEY_JSON = 'tweets_elnino_0125-0501'
    # KEY_JSON = 'tweets_lanina_0211-0429'

    tweet_dict = {}
    tweet_dict[KEY_JSON] = []

    # check every json entries one by one
    while len(temp_json[KEY_JSON]) != 0:
        poppy = temp_json[KEY_JSON].pop(0)
        tweet_temp = {}
        tweet_temp['text'] = poppy['text'].encode('utf-8')
        tweet_temp['urls'] = None
        tweet_temp['rt'] = 0
        tweet_dict[KEY_JSON].append(tweet_temp)

    with open(OUT_FILE,'w') as outFile:
        outFile.write(json.dumps(tweet_dict, indent = 4))


def extract_url(data_path, tweet_file):
    '''
    extract URLs from tweets
    '''

    # DATA_PATH = ''
    # TWEET_FILE = 'tweets_elnino_0125-0501_simple.json'
    # # TWEET_FILE = 'tweets_lanina_0211-0429_simple.json'
    # # OUT_FILE = 'tweets_elnino_0125-0501_simple.json'
    # # OUT_FILE = 'tweets_lanina_0211-0429_simple.json'
    # OUT_FILE = 'test_out_01.json'

    DATA_PATH = data_path
    TWEET_FILE = tweet_file
    OUT_FILE = tweet_file

    print "loading simple tweets file..."
    with open(DATA_PATH + TWEET_FILE, 'r') as inFile: 
        temp_json = json.load(inFile)
    print "loading simple tweets - done."

    # KEY_JSON = 'tweets_elnino_0125-0501'
    # KEY_JSON = 'tweets_lanina_0211-0429'

    # tweet_dict = {}
    # tweet_dict[KEY_JSON] = []
    tweet_list = []

    # initialize counters
    tweet_counter = 0
    timeout_error_counter = 0
    unknown_error_counter  = 0
    
    # check every json entries one by one
    # while len(temp_json[KEY_JSON]) != 0:
    # for i in range(len(temp_json[KEY_JSON])):
    # for i in range(tweet_counter,len(temp_json[KEY_JSON])):
    # for i in range(tweet_counter,tweet_counter+10000):
    for i in range(0,len(temp_json)):
        print "---------------------------"

        tweet_counter += 1
        # print str(tweet_counter) + "/" + str(len(temp_json[KEY_JSON]))
        print str(tweet_counter) + "/" + str(len(temp_json))

        # pop the first element of json
        # poppy = temp_json[KEY_JSON].pop(0)
        # poppy = temp_json[KEY_JSON][i]
        poppy = temp_json[i]

        tweet_temp = {}
        text = poppy['text'].encode('utf-8')
        tweet_temp['text'] = text

        urls = []

        try:
            # # find only the first URL
            # url = re.search("(?P<url>https?://[^\s]+)", text).group("url")

            # # find all URLs in the tweet
            urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

            print urls

        except AttributeError:
            pass


        # initialize a temporary list of unshortened URLs
        url_list = []
        # initialize URL counter for a tweet
        url_counter = 0

        for url in urls:
            url_counter += 1
            # print "--URL " +str(url_counter)

            # Start the timer. Once 5 seconds are over, a SIGALRM signal is sent.
            signal.alarm(4)    
            # this try/except loop ensures that you'll catch TimeoutException when it's sent.
            try:
                try:
                    print "(0) : "+url
                    # try:

                    if urlunshort.resolve(url) == None:
                        print "directly used"
                        url_list.append(url)
                    else:
                        try:
                            url = urlunshort.resolve(url)
                            print "(1) : "+url
                            if urlunshort.resolve(url) == None:
                                print "1 time resolved"
                                url_list.append(url)
                            else:
                                try:
                                    url = urlunshort.resolve(url)
                                    print "(2) : "+url
                                    print "2 times resolved"
                                    url_list.append(url)
                                    # if urlunshort.resolve(url) == None:
                                    #     print "2 times resolved"
                                    #     url_list.append(url)
                                    # else:
                                    #     try:
                                    #         url = urlunshort.resolve(url)
                                    #         print "(3) : "+url
                                    #         print "3 times resolved"
                                    #         url_list.append(url)
                                    #     except:
                                    #         print "#### Unknown ERROR found : level 3"
                                    #         unknown_error_counter += 1
                                    #         continue
                                except:
                                    print "#### Unknown ERROR found : level 2"
                                    unknown_error_counter += 1
                                    continue
                        except:
                            print "#### Unknown ERROR found : level 1"
                            unknown_error_counter += 1
                            continue
                except:
                    print "#### Unknown ERROR found : level 0"
                    unknown_error_counter += 1
                    continue

            except TimeoutException:
                print "#### TIMED OUT"
                timeout_error_counter += 1
                continue # continue the for loop if the function takes more than 5 second
            else:
                # seset the alarm
                signal.alarm(0)


        try:
            json.dumps(url_list)   # just to check UnicodeDecodeError on URLs
            if   len(url_list) != 0:
                tweet_temp['urls'] = url_list 
                print "URL(s) added"
                print url_list
            elif len(url_list) == 0:
                tweet_temp['urls'] = None
                print "NO URLs"
        except UnicodeDecodeError: # if URL gives UnicodeDecodeError, just ignore it
            tweet_temp['urls'] = None
            print "NO URLs"

        tweet_temp['rt']   = 0

        # tweet_dict[KEY_JSON].append(tweet_temp)
        tweet_list.append(tweet_temp)

    print str(timeout_error_counter) + " Timeout Errors"
    print str(unknown_error_counter) + " Unknown Errors"

    print "------------------------------------"
    print "outFile : " + OUT_FILE + "\n"
    print "------------------------------------"

    outFile = open(DATA_PATH+OUT_FILE,'w')
    # outFile.write(poppy['text'].encode('utf-8') + '\n')
    # outFile.write(json.dumps(tweet_dict, indent = 4))
    outFile.write(json.dumps(tweet_list, indent = 4))
    outFile.close()


def divide_tweets_db():
    '''
    divide a huge tweets json file into small ones
    '''

    DATA_PATH = ''
    TWEET_FILE = 'tweets_elnino_0125-0501_simple.json'
    OUT_FILE = 'tweets_elnino_0125-0501_simple'

    print "loading simple tweets file..."
    inFile = open(DATA_PATH + TWEET_FILE, 'r')
    temp_json = json.load(inFile)
    inFile.close()
    print "loading simple tweets - done."

    KEY_JSON = 'tweets_elnino_0125-0501'

    total_length = len(temp_json[KEY_JSON])
    total_step = total_length / 10000

    for i in range(total_step+1):
        outFile = open(DATA_PATH+OUT_FILE+'_'+str(i+1).zfill(3)+'.json','w')
        outFile.write(json.dumps(temp_json[KEY_JSON][0+10000*i:min(0+10000*(i+1),total_length)], indent = 4))
        outFile.close()


def combine_tweets_db():
    '''
    combine small tweets json files into one big json
    '''

    DATA_PATH = ''
    TWEET_FILE = 'tweets_elnino_0125-0501_simple'
    OUT_FILE = 'tweets_elnino_0125-0501_simple_urls.json'

    tweet_list = []
    print "loading simple tweets file..."
    for i in range(123):
        inFile = open(DATA_PATH + TWEET_FILE + '_' + str(i+1).zfill(3) + '.json', 'r')
        temp_json = json.load(inFile)
        inFile.close()
        tweet_list.extend(temp_json)
    print "loading simple tweets - done."

    KEY_JSON = 'tweets_elnino_0125-0501'

    outFile = open(DATA_PATH+OUT_FILE,'w')
    outFile.write(json.dumps(tweet_list, indent = 4))
    outFile.close()


def url_compare(url, num_char):
    '''
    getting resolved url address?? NEED TO CHECK
    '''
    return urlparse.urlparse(url).path[:num_char]


def count_tweets():
    '''
    counting number of tweets?? NEED TO CHECK
    '''

    # read simple_urls into json format
    DATA_PATH = ''
    TWEET_FILE = 'tweets_elnino_0125-0501_simple_urls.json'

    print "loading simple tweets file..."
    inFile = open(DATA_PATH + TWEET_FILE, 'r')
    tweet_urls = json.load(inFile)
    inFile.close()
    print "loading simple tweets - done."
    print "  total number of tweets  : " + str(len(tweet_urls))


    # read articles_db into json format
    DATA_PATH = 'data/'
    ARTICLE_FILE = 'articles_db.json'

    print "loading articles db file..."
    inFile = open(DATA_PATH + ARTICLE_FILE, 'r')
    articles_db = json.load(inFile)
    inFile.close()
    print "loading articles db - done."
    print "  total number of articles: " + str(len(articles_db))

    # find and count tweets/RTs and add it as "tweets"/"rt" fields
    print "finding URLs..."
    for i in range(len(articles_db)):

        url  = url_compare(articles_db[i]['url'],35 )
        tweet_count = 0
        rt_count    = 0

        for j in range(len(tweet_urls)):
            if tweet_urls[j]['urls'] != None:
                for k in range(len(tweet_urls[j]['urls'])):
                    if tweet_urls[j]['urls'][k] != None:
                        if url in tweet_urls[j]['urls'][k]:
                            tweet_count += 1
                            if 'RT' in tweet_urls[j]['text']:
                                rt_count += 1

        articles_db[i]['tweets'] = tweet_count
        articles_db[i]['rt'] = rt_count

        print str(i).zfill(3) + '_th article : ' + str(tweet_count) + '\t' + str(rt_count)

    # write to the original file
    DATA_PATH = 'data/'

    outFile = open(DATA_PATH+ARTICLE_FILE,'w')
    outFile.write(json.dumps(articles_db, indent = 4))
    outFile.close()


def get_n_closest(vector, N=5):

    n_closest = []

    VECTOR_FILE = 'data/glove.6B.50d.vector'
    VOCAB_FILE  = 'data/glove.6B.50d.vocab'

    with open(VECTOR_FILE,'r') as f_in:
        lines = f_in.readlines()

    # calculate similarities to all GloVe vectors
    simil_list = []
    for count, line in enumerate(lines):
        line_vec = np.asarray(line.strip('\n ').split(' '), dtype=np.float32)
        similarity = cosine_similarity(line_vec.reshape(1,-1), vector.reshape(1,-1))[0][0] # <- 왜 [0][0] 이지?
        simil_list.append(similarity)
        if count % 3000 == 0:
            progress_bar(count,len(lines))
    print "\n"

    # sort them and get indices for N closests            
    idx_n_closest = np.argsort(simil_list)[::-1][:N]

    # read vocabulary indices
    with open(VOCAB_FILE,'r') as f_in:
        lines = f_in.readlines()

    # get words for those N closest vectors
    for idx in idx_n_closest:
        word = lines[idx].strip('\n')
        n_closest.append(word)

    return n_closest 


def normalized(text, 
               remove_punc=False, 
               remove_stops=False, 
               singularize=False,
               stem="none", 
               lemmatize=False):

    stops = stopwords.words('english')
    stops.extend(['said','many','must','also']) # add some more words to nltk stopwords

    words = nltk.pos_tag(text.lower().split())

    if remove_punc:
        words = [(word[0].strip('"~`!?<>{}[]+=@#$%^&*.,:;-_)(/\'|'), word[1]) for word in words]
        words = [(word[0].replace("'s",""), word[1]) for word in words]

    if remove_stops:
        words = [word for word in words if word[0] not in stops]

    if singularize:
        words = [(inflection.singularize(word[0]), word[1]) for word in words]
    
    if stem != "none":
        if   stem == "porter":
            stemmer = PorterStemmer()  
        elif stem == "lancaster":
            stemmer = LancasterStemmer()  
        elif stem == "snowball":
            stemmer = nltk.SnowballStemmer("english")
        words = [(stemmer.stem(word[0]), word[1]) for word in words]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
        words = [(lemmatizer.lemmatize(word[0], pos_dict.get(word[1][0],wordnet.NOUN)), word[1]) for word in words]

    words = [word[0] for word in words]

    return (' ').join(words)


def load_glove_vectors(PATH, FILE):
    print "loading Word Embedding..."
    with open(PATH + FILE, 'r') as infile:
        gloveset = infile.readlines()
    itemized_glove = [item.split() for item in gloveset]
    glove_dict = {}
    for idx, vec in enumerate(itemized_glove):
        glove_dict[vec[0]] = np.asarray(vec[1:], dtype=np.float32) # cast as float 32, which is the format of GloVe
    print "done."

    return glove_dict


def word_emb_avg(text, word_embedding_dict):

    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word not in stopwords.words('english')]  # remove stopwords

    # average
    wdemb_stack = np.zeros(len(word_embedding_dict['the']))  # initialize with dummy
    for word in words:
        if word in word_embedding_dict:
            wdemb_stack = np.vstack((wdemb_stack, word_embedding_dict[word]))
    wdemb_stack = wdemb_stack[1:]   # remove first dummy
    wdemb_avg = np.mean(wdemb_stack.astype(np.float32), axis=0)

    return wdemb_avg


def evaluate(result, target, threshold, mode='binary', verbose=True):
    total_count = len(result)
    comparison = zip(result,target)

    # show prediction and target pairs
    if verbose:
        for i in range(total_count): print i, comparison[i]

    if   mode == 'binary':
        threshold_hotnot = threshold[0]
        result_hotnot = []
        target_hotnot = []
        hot_cnt = 0

        for res in result:
            if res > threshold_hotnot: result_hotnot.append('++')
            else: result_hotnot.append('__')
        
        for tgt in target:
            if tgt > threshold_hotnot:
                target_hotnot.append('++')
                hot_cnt += 1
            else:
                target_hotnot.append('__')

        comparison_hotnot = zip(result_hotnot, target_hotnot)

        print "Hot Ratio: {}".format(hot_cnt/total_count)

        # Complexity Matrix
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(total_count):
            # if verbose: print i, comparison_hotnot[i]
            if   comparison_hotnot[i][0] == '++' and comparison_hotnot[i][1] == '++': TP += 1
            elif comparison_hotnot[i][0] == '++' and comparison_hotnot[i][1] == '__': FP += 1
            elif comparison_hotnot[i][0] == '__' and comparison_hotnot[i][1] == '__': TN += 1
            elif comparison_hotnot[i][0] == '__' and comparison_hotnot[i][1] == '++': FN += 1

        acc = (TP+TN)/total_count
        prc = TP/(TP+FP)
        rec = TP/(TP+FN)
        f1  = 2*prc*rec/(prc+rec)
        
        if verbose :
            print "TP: {}, FP: {}, TN: {}, FN: {}".format(TP, FP, TN, FN)
            print "Accuracy : {0:.2f}".format(acc)
            print "Precision: {0:.2f}".format(prc)
            print "Recall   : {0:.2f}".format(rec)
            print "F1 score : {0:.2f}".format(f1)

        return acc, prc, rec, f1


    elif mode == 'ternary':
        threshold_upper = max(threshold)
        threshold_lower = min(threshold)
        result_hml = []
        target_hml = []
        high_cnt, mid_cnt, low_cnt = 0, 0, 0

        for res in result:
            if   res > threshold_upper: result_hml.append('++')
            elif res > threshold_lower: result_hml.append('//')
            else: result_hml.append('__')
        
        for tgt in target:
            if   tgt > threshold_upper:
                target_hml.append('++')
                high_cnt += 1
            elif tgt > threshold_lower:
                target_hml.append('//')
                mid_cnt += 1
            else:
                target_hml.append('__')
                low_cnt += 1

        comparison_hml = zip(result_hml, target_hml)

        print "High Ratio: {0:.2f}".format(high_cnt/total_count)
        print "Mid  Ratio: {0:.2f}".format(mid_cnt/total_count)
        print "Low  Ratio: {0:.2f}".format(low_cnt/total_count)


        # something like Complexity Matrix
        TH, FH, TM, FM, TL, FL = 0,0,0,0,0,0
        for i in range(total_count):
            # if verbose: print i, comparison_hml[i]
            if   comparison_hml[i][0] == '++':
                if   comparison_hml[i][1] == '++': TH += 1
                else : FH += 1
            elif comparison_hml[i][0] == '//':
                if   comparison_hml[i][1] == '//': TM += 1
                else : FM += 1
            elif comparison_hml[i][0] == '__':
                if   comparison_hml[i][1] == '__': TL += 1
                else : FL += 1

        acc = (TH+TM+TL)/total_count
        _   = None
        
        if verbose :
            print "TH: {}, FH: {}, TM: {}, FM: {}, TL: {}, FL: {}".format(TH, FH, TM, FM, TL, FL)
            print "Accuracy : {0:.2f}".format(acc)

        return acc, _, _, _

    elif mode  == 'regression':
        
        # percentage = [int(100*pair[0]/pair[1]) for pair in comparison]
        
        # if verbose: print percentage

        # acc = np.mean(percentage)
        acc = 0
        _   = None

        # print "Average (percentage) accuracy: {0:.2f}".format(np.mean(percentage))

        return acc, _, _, _






