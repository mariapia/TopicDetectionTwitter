import csv
import pandas
import sys
#from fpgrowth import *
import fpgrowth
from ast import literal_eval
import re


f_text = sys.argv[1] + '/codeOutput/user_topic_02.csv'

df = pandas.read_csv(f_text, delimiter=',', header=0, engine='python', escapechar='\\',
                     names=["u_id","topics"])

df.topics = df.topics.apply(literal_eval)


items = []
tmp = []
for row in df.iterrows():
    topics = row[1].get(1)
    basket = []
    for t in topics:
        basket.append(t)
    items.append(basket)

print(items)

frequent_sets = fpgrowth.frequent_itemsets(items, 5)

itemsets = dict(frequent_sets)
print(len(itemsets))
rules = fpgrowth.association_rules(itemsets, .8)
rules = list(rules)

clean_rules = []
tmp_rule = []
regex = r"{(.*?)}"
for rule in rules:
    matches = re.finditer(regex, str(rule))
    for matchnum, match in enumerate(matches):
        r = match.group(1).replace("'", "")
        tmp_rule.append(r)
    clean_rules.append(tmp_rule)
    tmp_rule = []


print(clean_rules)

with open(sys.argv[1]+"/codeOutput/frequent_itemsets_02.csv", 'w') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    for rule in clean_rules:
        wr.writerow(rule)
