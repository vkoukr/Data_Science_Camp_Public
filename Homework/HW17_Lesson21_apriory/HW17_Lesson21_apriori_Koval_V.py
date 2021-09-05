
# !pip install apyori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

help(apriori)

df = pd.read_csv('Groceries_dataset.csv')
df.head ()
len(df)
dist = [len(g) for k, g in df.groupby(['Member_number', 'Date'])]
plt.hist(dist)

products = df["itemDescription"].unique()

#---Transaction selection
df_dataprep=df.groupby(['Member_number','Date'])['itemDescription'].apply(','.join).reset_index()
transactions=df_dataprep[['itemDescription']].values
trans=[(''.join(i).split(",")) for i in transactions]
print(trans)

# Rule selection
rules = list(apriori(trans, min_support = 0.003,  min_confidence = 0.01, min_lift = 1.01, min_length = 2))
print(rules)
print(len(rules))

# Print rules in more details
def print_rules(rules):
    for rule in rules:
        print ('rule.items=', list(rule.items))
        print ('rule.support=',rule.support)

        for os in rule.ordered_statistics:
            print ('\titems_base=', list(os.items_base))
            print ('\tlifted_item =', list(os.items_add))
            print ('\tlift=', os.lift)
            print ('\tconfidence (i.e. cond prob {} if {})='.format(list(os.items_add), list(os.items_base)), os.confidence)
            print ('\t----')
        print ('\n')
        
print_rules(rules)