import csv
import pandas
import sys
import numpy as np
import fpgrowth
from ast import literal_eval

f_text = sys.argv[1] + '/codeOutput/complete_02.csv'


df = pandas.read_csv(f_text, delimiter=',', header=0, engine='python', escapechar='\\',
                     names=["t_id", "u_id", "text", "topics", "tag_id", "hashtag", "tokens"])

df.topics = df.topics.apply(literal_eval)
df.hashtag = df.hashtag.apply(literal_eval)

# group users by id
df_user_topics = df.groupby(['u_id'])['topics'].apply(list)

df_user_hashtag = df.groupby(['u_id'])['hashtag'].apply(list)

topic_df_grouped = df_user_topics.to_frame().merge(df_user_hashtag.to_frame(), how='left', on=['u_id'],
                                                   right_index=False).apply(list)

columns = ['u_id', 'topics']
df_user_topic = pandas.DataFrame(columns=columns)

topic_users_df = pandas.DataFrame()

items = []
tmp = []
c = 0

for row in topic_df_grouped.iterrows():
    topic_user = []
    topics_list = row[1].get(0)

    for single_list in topics_list:
        if len(single_list) > 0:
            if single_list[0].isdigit() == False and single_list[1].isdigit():
                topic_user.append(single_list[0])
            elif single_list[0].isdigit and single_list[1].isdigit == False:
                topic_user.append(single_list[1])
            else:
                topic_user.append(single_list[0])
                topic_user.append(single_list[1])

    hashtag_list = row[1].get(1)

    if len(hashtag_list) > 0:
        for single_list in hashtag_list:
            if len(single_list) > 0:
                for hg in single_list:
                    print(hg)
                    topic_user.append(hg)

    if topic_user:
        topic_user = list(set(topic_user))
        items.append(topic_user)
        df_user_topic = df_user_topic.append(
            {'u_id': row[0], 'topics': topic_user}, ignore_index=True)

        dic = {}
        dic['u_id'] = row[0]
        counter = 1
        for t in topic_user:
            dic['topic'+str(counter)] = t
            counter+=1

        topic_users_df = topic_users_df.append(dic, ignore_index=True)

df_user_topic.to_csv(path_or_buf=sys.argv[1] + '/codeOutput/user_topic_02.csv', sep=',', na_rep='', header=0, index=False,
                     index_label=None, mode='w', doublequote=True, escapechar='\\')

topic_users_df.to_csv(path_or_buf=sys.argv[1] + '/codeOutput/user_column_topics_02.csv', sep=',', na_rep='', header=True, index=False,
                     index_label=None, mode='w', doublequote=True, escapechar='\\')


columns = ['u_id', 'topic', 't_id']
df_user_duplicate = pandas.DataFrame(columns=columns)

for row in df.iterrows():
    topics = list(set(row[1].topics))
    hashtag = list(set(row[1].hashtag))
    total_topics = list(set(topics + hashtag))
    for t in total_topics:
        dt = {}
        dt['u_id'] = row[1].u_id
        dt['topic'] = t
        dt['t_id'] =row[1].t_id
        df_user_duplicate = df_user_duplicate.append(dt, ignore_index=True)
        print(dt)

    print("----------------------------------")


df_userTopic_grouped = df_user_duplicate.groupby(['u_id', 'topic'], as_index=False).count()

print(df_userTopic_grouped)

df_userTopic_grouped.to_csv(path_or_buf=sys.argv[1] + '/statistics/userTopic_grouped_02.csv', sep=',', na_rep='', header=True, index=False,
                     index_label=None, mode='w', doublequote=True, escapechar='\\')
