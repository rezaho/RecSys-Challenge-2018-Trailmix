import math
import json
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
from time import time

# from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_trainRatings = None
_all_item_idices = None
_all_item_idices1 = None

def evaluate_model(model, all_item_idices, trainRatings, all_item_idices1):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _trainRatings
    global _all_item_idices
    global _all_item_idices1
    global _all_item_idices2

    _model = model
    _trainRatings = trainRatings
    _all_item_idices = all_item_idices
    _all_item_idices1 = all_item_idices1

    i = 0
    t1 = time()
    Final = {}

    print('QQ: Start_read_music_similar')
    music_similar = json.load(open('word2vec_similar_song/word2vec_all_song_100.json'))
    print('QQ: Finish_read_music_similar')
    t2 = time()
    print('Read_Time:')
    print(t2-t1)

    t1 = time()
    for u in _trainRatings.keys():
        i += 1
        if i % 100 == 0:
            t2 = time()
            print('This is:')
            print(i)
            print('Finish!')
            print(t2 - t1)
            t1 = time()
        if len(_trainRatings[u]) != 0:
            items3 = []
            for j in _trainRatings[u]:
                items3 = items3 + music_similar[str(j)][0:50]

            items3 = set(items3)
            Final[u] = eval_one_list(u, items3)

    return Final




def eval_one_list(idx, items3):
    u = idx  # _test_list_idices[idx]


    # items = range(_num_items)
    items1 = _all_item_idices[u][0:500]
    items2 = list(_all_item_idices1)
    # items3 = _all_item_idices2[u]

    tmp = set(_trainRatings[u])
    items2 = list(set(items2) - tmp)

    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items2), int(u), dtype='int32')
    predictions = _model.predict([users, np.array(items2)],
                                 batch_size=100, verbose=0)
    for i in xrange(len(items2)):
        item2 = items2[i]
        map_item_score[item2] = predictions[i]
# items.pop()
    ranklist1 = heapq.nlargest(500, map_item_score, key=map_item_score.get)

    ranklist = []
    t = 0
    for i in range(500):
        if (ranklist1[i] in items1) and (ranklist1[i] in items3) and (t < 500):
            ranklist.append(ranklist1[i])
            t += 1

    # print('xing')
    # print(t)
    # print('xing')

    if t < 500:
        for i in ranklist1:
            if (i not in ranklist) and (t < 500) and (i in items1):
                ranklist.append(i)
                t += 1

    if t < 500:
        for i in ranklist1:
            if (i not in ranklist) and (t < 500) and (i in items3):
                ranklist.append(i)
                t += 1

    if t < 500:
        for i in ranklist1:
            if (i not in ranklist) and (t < 500):
                ranklist.append(i)
                t += 1

    return ranklist


# G is grand truth, R is predicted (lenth must be 500 and will not have any song in existed ground truth). Type(G) == Type(R) == list
def r_precision(G, R):
    limit_R = R[:len(G)]
    if len(G) != 0:
        return len(list(set(G).intersection(set(limit_R)))) * 1.0 / len(G)
    else:
        return 0

def F_ndcg(G, R):
    r = [1 if i in set(G) else 0 for i in R]
    r = np.asfarray(r)
    dcg = r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    #k = len(set(G).intersection(set(R)))
    k = len(G)
    if k > 0:
        idcg = 1 + np.sum(np.ones(k - 1) / np.log2(np.arange(2, k + 1)))
        return dcg * 1.0 / idcg
    else:
        return 0


# def clicks(G, R):
#     r = [1 if i in set(G).intersection(set(R)) else 0 for i in R]
#     if sum(r) == 0:
#         return 51
#     else:
#         return int((r.index(1)) * 1.0 / 10)


def clicks(G, R):
    n = 1
    for i in R:
        if i in set(G):
            # return (n - 1) * 1.0 / 10
            return ((n-1)/ 10) * 1.0
        n += 1
    return 51