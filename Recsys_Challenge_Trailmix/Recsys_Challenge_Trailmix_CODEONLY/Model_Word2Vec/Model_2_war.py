import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import codecs
import json
import gensim
import numpy as np
import math
import random
import numpy
import operator
import scipy
import Queue
from heapq import nlargest
import sys

start = int(sys.argv[1]) * 500
end = start + 500

# Calculation Tools
def find_nearest_k(pid, k):
    global PL_Word2Vec_TEST
    global TEST
    
    res = []
    sim = {}
    for track in model.wv.vocab.keys():
        sim[track] = 1 - scipy.spatial.distance.cosine(np.array(PL_Word2Vec_TEST[pid]),np.array(model.wv[track]))

    switch_sim = dict((k, v) for v, k in sim.items()) 
    sorted_sim = sorted(switch_sim.keys(), reverse = True)
    for i in range(k):
        res.append(switch_sim[sorted_sim[i]])

    return res

# Load Model (MUST USE LATEST ONE)
model = gensim.models.Word2Vec.load('song2vec_GT5_TRAIN')

# Calculate the average vector
track_sum_vec = np.zeros(100, dtype = float)
for track in model.wv.vocab:
    track_sum_vec += model.wv[track]

track_average_vec = track_sum_vec / len(model.wv.vocab)

all_test_tracks = model.wv.vocab.keys()

TEST = json.load(open('WAR_PL_TRACKS_READONLY.json'))



# Word2Vec for PL

PL_Word2Vec_TEST = {}
for pl in TEST.keys()[start:end]:
    current = np.zeros(100)
    length = 0
    if len(TEST[pl]) == 0:
        PL_Word2Vec_TEST[pl] = list(track_average_vec)
    else:
        for track in TEST[pl]:
            if track in model.wv:
                current += model.wv[track]
            else:
                current += track_average_vec
            length += 1
        
        if length == 0:
            PL_Word2Vec_TEST[pl] = list(track_average_vec)
        else:
            PL_Word2Vec_TEST[pl] = list(current * 1.0 / length)

Predicted = {}
count_ = start
for pl in TEST.keys()[start:end]:
    print count_
    count_ += 1
    Predicted[pl] = find_nearest_k(pl, 500)

filename = 'WAR_PREDICT/WAR_Predict_' + sys.argv[1] + '.json'
with open(filename, 'w') as fp:
    json.dump(Predicted, fp)