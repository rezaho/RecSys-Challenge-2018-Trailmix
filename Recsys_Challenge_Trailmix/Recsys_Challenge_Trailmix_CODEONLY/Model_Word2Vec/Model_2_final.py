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

# Evaluation Metircs
def r_precision(G, R):
    limit_R = R[:len(G)]
    if len(G) != 0:
        return len(list(set(G).intersection(set(limit_R)))) * 1.0 / len(G)
    else:
        return 0

def ndcg(G, R):
    r = [1 if i in set(G).intersection(set(R)) else 0 for i in R]
    r = np.asfarray(r)
    dcg = r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    #k = len(set(G).intersection(set(R)))
    k = len(G)
    if k > 0:
        idcg = 1 + np.sum(np.ones(k - 1) / np.log2(np.arange(2, k + 1)))
        return dcg * 1.0 / idcg
    else:
        return 0

def clicks(G, R):
    r = [1 if i in set(G).intersection(set(R)) else 0 for i in R]
    if sum(r) == 0:
        return 51
    else:
        return (r.index(1) - 1) * 1.0 / 10 

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

# Recommendation
for task in range(10):
	print "Starting TASK " + str(task)

	print "Loading Data..."
	TEST = json.load(open('../DATA_PROCESSING/PL_TRACKS_5_TEST_T' + str(task) + '.json'))

	SUB_TEST_PL = TEST.keys()[:500]
	SUB_TEST = {k:v for (k,v) in TEST.items() if k in SUB_TEST_PL} 

	# Load Ground Truth
	TEST_GROUND_TRUTH_RAW = json.load(open('../DATA_PROCESSING/PL_TRACKS_5_TEST.json'))
	TEST_GROUND_TRUTH = {}
	for pl in SUB_TEST:
	    TEST_GROUND_TRUTH[pl] = TEST_GROUND_TRUTH_RAW[pl]

	# Word2Vec for PL

	PL_Word2Vec_TEST = {}
	for pl in SUB_TEST:
		current = np.zeros(100)
		length = 0
		for track in SUB_TEST[pl]:
			if track in model.wv:
				current += model.wv[track]
			else:
				current += track_average_vec
			length += 1
		if length != 0:
			PL_Word2Vec_TEST[pl] = list(current / length)
		else:
			PL_Word2Vec_TEST[pl] = track_average_vec

	print 'Loading Ground Truth...'

	R = {}
	G = {}

	for pl in TEST_GROUND_TRUTH:
		G[pl] = list(set(TEST_GROUND_TRUTH[pl]).difference(set(SUB_TEST[pl])))

	print 'Finding Nearest K Neighbors...'
	for pl in SUB_TEST:
		R[pl] = find_nearest_k(pl, 500)

	print 'Evaluating...'
	print 'R_Precision:'

	

	r_pre_result = []
	for pl in SUB_TEST.keys():
		r_pre_result.append(r_precision(G[pl], R[pl]))
	r_pre_result = np.array(r_pre_result)

	print '\tmean:', r_pre_result.mean(), 'std:', r_pre_result.std()

	with open('Model_2_Performance_500.txt', 'a') as f:
	    f.write('TASK_' + str(task) +' R_Precision_Mean '+str(r_pre_result.mean())+' R_Precision_Std '+ str(r_pre_result.std()) + '\n')


	print 'NDCG:'
	ndcg_result = []
	for pl in SUB_TEST.keys():
		ndcg_result.append(ndcg(G[pl], R[pl]))
	ndcg_result = np.array(ndcg_result)
	print '\tmean:', ndcg_result.mean(), 'std:', ndcg_result.std()

	with open('Model_2_Performance_500.txt', 'a') as f:
	    f.write('TASK_' + str(task) +' NDCG_Mean '+str(ndcg_result.mean())+' NDCG_Std '+ str(ndcg_result.std()) + '\n')


	print 'Clicks:'
	clicks_result = []
	for pl in SUB_TEST.keys():
		clicks_result.append(clicks(G[pl], R[pl]))
	clicks_result = np.array(clicks_result)
	print '\tmean:', clicks_result.mean(), 'std:', clicks_result.std()

	with open('Model_2_Performance_500.txt', 'a') as f:
	    f.write('TASK_' + str(task) +' Clicks_Mean '+str(clicks_result.mean())+' Clicks_Std '+ str(clicks_result.std()) + '\n')

	print '========================================\n'
	
