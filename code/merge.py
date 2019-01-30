import pandas
import csv
import sys

f_text = sys.argv[1] + '/tweet_text_02.csv'
f_ht = sys.argv[1] + '/tweet_hashtag_02.csv'

df1 = pandas.read_csv(f_ht, delimiter=',', header=None, usecols=[0,2], names=["tweet_id", "tag_id"])
df2 = pandas.read_csv(f_text, delimiter=',', header=None, engine='python', escapechar='\\', usecols=[0,1,2], names=["tweet_id", "user_id", "text", "geo_lat", "geo_long", "place_full_name", "place_id"])

df1_grouped = df1.groupby(['tweet_id'])['tag_id'].apply(list)
df = df2.merge(df1_grouped.to_frame(), how='left', on=['tweet_id'], right_index=False)

df.columns = ['t_id', 'u_id', 'text', 'tag_id']

df.to_csv(path_or_buf=sys.argv[1] +'codeOutput/text_ht_02.csv', sep=',', na_rep='"', header=True, index=False, index_label=None, mode='w', doublequote=True, escapechar='\\')
