import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import string
import re
import operator

from time import time
from pylab import plot,show
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
input_file = open("train_4.csv",'r')
reader = csv.reader( input_file )
reader.next()
Text = []
Tags = [] 

i = 0

for line in reader:
    Text.append(str(line[1] + " " + line[2]))
    Tags.append(str(line[3]))
    i += 1
    if (i == 2000):
        break

input_file.close()
Y_vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), min_df=0, binary=True)
Y = Y_vectorizer.fit_transform(Tags)
tags = Y_vectorizer.get_feature_names()

num_posts, num_words = Y.shape
print 'number of tags in dataset = %s' % num_words
counts = Y.sum(axis=0).tolist()[0]
word_counts = list(enumerate(counts))

word_counts_sorted = sorted(word_counts, key=lambda x: x[1], reverse=True)
word_counts_sorted = map(lambda x: (tags[x[0]], float(x[1])/num_posts), word_counts_sorted)

words_sorted, counts_sorted = zip(*word_counts_sorted)
N = 40
ind = np.arange(N)    # the x locations for the groups
width = 0.4       # the width of the bars: can also be len(x) sequence
plt.figure()

p1ot = plt.bar(ind, counts_sorted[:N],  width)
plt.title('Tag Frequency in Train Set')
plt.xlabel('Top %s Tags' % N)
plt.ylabel('Frequency') 
plt.xticks(ind+width/2., words_sorted[:N], rotation=90 )
plt.yticks(np.arange(0,.1,.01))
plt.show()
