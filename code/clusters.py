import pandas as pd
import sys
import re

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans
from sklearn.externals import joblib

from ast import literal_eval

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

#f_text = sys.argv[1]+'complete_5000.csv'
#u_text = sys.argv[1]+'user_topic_5000.csv'

f_text = sys.argv[1]+'/codeOutput/complete_02.csv'
u_text = sys.argv[1]+'/codeOutput/user_topic_02.csv'
doc = sys.argv[1] + '/twitter-stopwords3.txt'

sw = []
for line in open(doc):
    for word in line.split(","):
        sw.append(word)

all_stops = stop.union(set(sw))

df1 = pd.read_csv(f_text, delimiter=',',usecols=[0,1,2], names=['t_id','u_id', 'text'])
df2 = pd.read_csv(u_text, delimiter=',',  names=['u_id', 'topics'])



df2.topics = df2.topics.apply(literal_eval)
df = df2.merge(df1,  how='left', on=['u_id'], right_index=False)
df.to_csv(path_or_buf=sys.argv[1] + '/codeOutput/tmp_merged_02.csv', sep=',', na_rep='"', header=False, index=False, index_label=None, mode='w', doublequote=True, escapechar='\\')


tweet_id = list(df.t_id)
users = list(df.u_id)
texts= list(df.text)
topics = list(df.topics)


indexes = []
ranks = []
words = []

for i in range(0,len(tweet_id)):
    ranks.append(i)

def preprocess_tokenize(sentence):
    sentence = re.sub('http[s]*:\/\/[a-zA-Z0-9\.]+\/[a-zA-Z0-9]+', '', sentence)
    sentence  = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in all_stops]
    return filtered_words

def preprocess_lemmatization(sentence):
    sentence = re.sub('http[s]*:\/\/[a-zA-Z0-9\.]+\/[a-zA-Z0-9]+', '', sentence)
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [lemma.lemmatize(w) for w in tokens if not w in all_stops]
    return filtered_words


for i in texts:
    allwords_tokenized = preprocess_tokenize(i)
    words.extend(allwords_tokenized)

    allwords_lemmatized = preprocess_lemmatization(i)  # for each item in 'synopses', tokenize/stem
    indexes.extend(allwords_lemmatized)  # extend the 'totalvocab_stemmed' list



print("length tokens "+str(len(words))+" , lenght index "+str(len(indexes)))

vocab_frame = pd.DataFrame({'words': words}, index = indexes)


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.02, stop_words='english',
                                 use_idf=True, tokenizer=preprocess_lemmatization, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(texts) #fit the vectorizer to synopses
print(tfidf_matrix)
print()
print()
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
print()
print(terms)

dist = 1 - cosine_similarity(tfidf_matrix)

## ----- k-means clustering ----- ##

num_clusters = 10
km = KMeans(n_clusters=num_clusters, random_state=0)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

#to save the results of the cluster and load it again
joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()


print("Top terms per cluster:")
print()
# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
clusters_legend = []
legend_df = pd.DataFrame()


for i in range(num_clusters):
    d={}
    d['index'] = i
    print("Cluster %d words:" % i, end='')
    string = ""
    for ind in order_centroids[i, :5]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')

        string = string + str(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0]) + ", "
        d['w'+str(ind)] = string
        string=""
    clusters_legend.append(string)
    legend_df = legend_df.append(d, ignore_index=True)


legend_df.to_csv(path_or_buf=sys.argv[1] + '/codeOutput/legend_cluster_02.csv', sep=',', na_rep='', index=False, index_label=None, mode='w', doublequote=True, escapechar='\\')

tweets =  {'tweet_id':tweet_id, 'user_id':users,'topics':topics,'cluster': clusters }

frame = pd.DataFrame(tweets, index = [clusters] , columns = ['tweet_id','user_id', 'topics', 'cluster'])

frame.to_csv(path_or_buf=sys.argv[1] + '/codeOutput/tweets_cluster_02.csv', sep=',', na_rep='', index=False, index_label=None, mode='w', doublequote=True, escapechar='\\')


grouped = frame.groupby(['user_id'], as_index=False).cluster.agg({'clusters ':list})
userCsv = pd.DataFrame()

for row in grouped.iterrows():
    d={}
    d['user_id'] = row[1].user_id
    count = 1
    clusters = row[1].get(1)

    for c in clusters:
        d['cluster'+str(count)] = c
        count+=1

    userCsv = userCsv.append(d, ignore_index=True)

userCsv.to_csv(path_or_buf=sys.argv[1] + '/codeOutput/user_cluster_02csv', sep=',', na_rep='', index=False, index_label=None, mode='w', doublequote=True, escapechar='\\')


