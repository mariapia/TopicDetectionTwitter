import pandas
import csv
import sys
import string
import re
from ast import literal_eval

f_text = sys.argv[1] +'/codeOutput/user_cluster_02.csv'
u_text = sys.argv[1] +'/codeOutput/user_topic_02.csv'

df = pandas.read_csv(f_text, delimiter=',', header=1, engine='python', escapechar='\\', usecols=[0,1], names=["cluster1", "u_id"])
df_user = pandas.read_csv(u_text, delimiter=',', header=None, engine='python', escapechar='\\', names=["u_id", "topics"])

df_user.topics = df_user.topics.apply(literal_eval)

df.columns = ['cluster', 'u_id']

df_merged = df.merge(df_user, how='left', on=['u_id'], right_index=False)


df_groupedByCluster = df_merged.groupby(['cluster'], as_index=False).agg({'topics':list, 'u_id':list})

df_groupedByCluster.to_csv(path_or_buf=sys.argv[1] + '/codeOutput/division_by_cluster_02.csv', sep=',', na_rep='', header=None, index=False,
               index_label=None, mode='w', doublequote=True, escapechar='\\')

df_recomCluster = pandas.DataFrame(columns=['cluster','users', 'topics', 'n_users', 'n_topics'])

for elem in df_groupedByCluster.iterrows():
    d={}
    cluster=elem[1].cluster
    users = elem[1].u_id
    topics_cluster = elem[1].topics
    all_topics_cluster = []
    all_users_cluster = []
    for topic in topics_cluster:
        for t in topic:

            all_topics_cluster.append(t)
    size_topics =len(all_topics_cluster)
    for user in users:
        all_users_cluster.append(int(user))
    size_users = len(all_users_cluster)

    df_recomCluster = df_recomCluster.append({'cluster':cluster,'users':all_users_cluster ,'n_users': size_users, 'topics':all_topics_cluster, 'n_topics': size_topics}, ignore_index=True)


df_recomCluster.to_csv(path_or_buf=sys.argv[1] + '/statistics/rec_topics_cluster_02.csv', sep=',', na_rep='', header=1, index=False,
               index_label=None, mode='w', doublequote=True, escapechar='\\')

