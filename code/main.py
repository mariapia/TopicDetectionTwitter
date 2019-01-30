import pandas
import csv
import sys
import string
import re
import numpy
import fpgrowth


from functools import reduce

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer


from nltk import pos_tag
from nltk.stem.regexp import RegexpStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.isri import ISRIStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.rslp import RSLPStemmer

import gensim
from gensim import corpora

f_text = sys.argv[1] +'codeOutput/text_ht_02.csv'
doc = sys.argv[1] + '/twitter-stopwords3.txt'

sw = []
for line in open(doc):
    for word in line.split(","):
        sw.append(word)

d = dict([("''","n"),("$","n"),("CC", "n"),("CD", "n"),("DT", "n"),("EX", "n"),("FW", "n"),("IN", "n"),("JJ", "a"),("JJR", "a"),("JJS", "a"),("LS", "n"),("MD", "n"),("NN", "n"),("NNP", "n"),("NNPS", "n"),("NNS", "n"),("PDT", "n"),("POS", "n"),("PRP", "n"),("PRP$", "n"),("RB", "r"),("RBR", "r"),("RBS", "r"),("RP", "n"),("SYM", "n"),("TO", "n"),("UH", "n"),("VB", "v"),("VBD", "v"),("VBG", "v"),("VBN", "v"),("VBP", "v"),("VBZ", "v"),("WDT", "n"),("WP", "n"),("WP$", "n"),("WRB", "n")])

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
Lda = gensim.models.ldamodel.LdaModel

all_stops = stop.union(set(sw))

# text - testo iniziale lemmatizzato senza hashtags, mentions e urls
# lda - testo iniziale lemmatizzato senza hashtags, mentions e urls e senza stopwords + check su digits e lenght
# hashtag - testo iniziale lemmatizzato senza mentions e urls e senza stopwords + check su digits e lenght

#comune a tutti, lemmatizazione , remove di mentions e urls

def clean(sentence):
    sentence = sentence.lower()
    sentence = re.sub('http[s]*:\/\/[a-zA-Z0-9\.]+\/[a-zA-Z0-9]+', '', sentence)
    sentence = re.sub('@[a-zA-Z0-9_]+', '', sentence)
    return sentence

# lemmatizzazione per
def tokenize(sentence, ht,cc):
    sentence = re.sub('#([a-zA-Z0-9]+)', '\g<1>', sentence)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    tokens = pos_tag(tokens)
    tokens = [lemma.lemmatize(word[0], pos=d[word[1]]) for word in tokens if len(lemma.lemmatize(word[0], pos=d[word[1]]))>1]
    return tokens

def for_toks(tokens):
    filtered_words = [w for w in tokens if (len(w)>1 and (not w in all_stops)) and (not (w.isdigit() and len(w)>4))]
    return list(set(filtered_words))

def inline(tokens):
    inline = ' '.join(word for word in tokens)
    return inline


def preprocess(sentence, ht,cc):
    sentence = clean(str(sentence))
    
    if ht:
        ht = re.findall(r"#(\w+)", sentence)
        hashtag = []
        for t in ht:
            if not (re.match('((un)*fol+o[a-z]*|f4f|rt|retweet|mgwv|rt2gain)', t)):
                hashtag.append(t)

    wt = tokenize(sentence, ht,cc)

    if ht:
        return inline(wt), for_toks(wt), hashtag
    return inline(wt), for_toks(wt)

df = pandas.read_csv(f_text, delimiter=',', header=0, escapechar='\\', iterator=True, chunksize=1)

columns = ['t_id', 'u_id', 'text','topics', 'tag_id', 'hashtag']
topic_df = pandas.DataFrame(columns=columns)
topics_per_tweet = pandas.DataFrame()
max_number_topics = 0
cc = 0
for row in df:
    text = row['text'].get(cc)
    results = []
    hashtags = []

    flag = not (row['tag_id'].get(cc)=="\"")

    if flag:
        results = preprocess(text, flag, cc)
        text = results[0]
        doc_clean = [results[1]]
        if (len(results) > 2):
            hashtags = results[2]
    else:
        results = preprocess(text, flag, cc)
        text = results[0]
        doc_clean = [results[1]]
    
    if doc_clean[0].__len__() != 0:
        dictionary = corpora.Dictionary(doc_clean)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        
        ldamodel = Lda(doc_term_matrix, num_topics=2, id2word=dictionary, passes=50, random_state=1)

        list_t_topic = []
        for i in ldamodel.show_topics(num_topics=2, num_words=1, log=False, formatted=True):
            t = re.sub(r'[0-9]*\.[0-9]*\*\"(.*)\"', r'\1', i[1])
            list_t_topic.append(t)

        topic_df = topic_df.append(
            {'t_id': row['t_id'].get(cc), 'u_id': row['u_id'].get(cc), 'text':text, 'tokens': doc_clean,'topics': list_t_topic, 
            'tag_id': row['tag_id'].get(cc), 'hashtag':hashtags}, ignore_index=True)


        total_topics = list_t_topic + hashtags
        total_topics = list(set(total_topics))
        if max_number_topics < total_topics.__len__():
            max_number_topics = total_topics.__len__()

        dic_tw = {}
        dic_tw['t_id'] = row['t_id'].get(cc)
        counter = 0;

        number_of_tokens = doc_clean[0].__len__()
        dic_tw['n_word_text'] = number_of_tokens
        dic_tw['text'] = text
        for t in total_topics:
            print(t)
            dic_tw['topic' + str(counter)] = t
            counter += 1

        print(dic_tw)
        topics_per_tweet = topics_per_tweet.append(dic_tw, ignore_index=True)

    cc = cc + 1

topic_df.to_csv(path_or_buf=sys.argv[1] + 'codeOutput/complete_02.csv', sep=',', na_rep='', header=0, index=False,
                index_label=None, mode='w', doublequote=True, escapechar='\\')

topics_per_tweet.to_csv(path_or_buf=sys.argv[1] + 'codeOutput/tweets_column_topics_02.csv', sep=',', na_rep='', header=0, index=False,
                        index_label=None, mode='w', doublequote=True, escapechar='\\')


