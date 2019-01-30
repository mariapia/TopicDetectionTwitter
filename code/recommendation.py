import csv
import pandas
import sys
import fpgrowth
from ast import literal_eval
import numpy
import math
import re

f_text = sys.argv[1] + 'codeOutput/user_topic_02.csv'
fi_text = sys.argv[1] + '/codeOutput/frequent_itemsets_02.csv'

df = pandas.read_csv(f_text, delimiter=',', header=None, engine='python', escapechar='\\',
                     names=["u_id", "topics"], iterator=True, chunksize=1)

columns = ['u_id', 'actual_topic', 'recom_topic']
df_recommended = pandas.DataFrame(columns=columns)

tmp = []
topics_csv = dict()
with open(fi_text, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for r in reader:
        tmp = list(r)
        topics_csv[tmp[0]] = tmp[1]
print(topics_csv)

cc = 0
regex = r"'(.*?)'"

for row in df:
    values = []
    topics = []
    new_topics = []
    if not pandas.isnull(row.topics.get(cc)):
        matches = re.finditer(regex, row.topics.get(cc))
        for matchnum, match in enumerate(matches):
            topics.append(match.group(1))
            topic = match.group(1)
            if topic in topics_csv.keys():
                values.append(topics_csv[topic])
        for v in values:
            if v not in topics:
                new_topics.append(v)

    df_recommended = df_recommended.append({'u_id': row.u_id.get(cc), 'actual_topic': topics, 'recom_topic': new_topics}, ignore_index=True)
    cc +=1

df_recommended.to_csv(path_or_buf=sys.argv[1] + '/codeOutput/recommendation_02.csv', sep=',', na_rep='', header=False, index=False, index_label=None, mode='w', doublequote=True, escapechar='\\')
