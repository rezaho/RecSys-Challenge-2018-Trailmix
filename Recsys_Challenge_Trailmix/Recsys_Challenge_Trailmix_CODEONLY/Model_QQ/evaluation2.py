import heapq  # for retrieval topK
import numpy as np
from time import time

# Global variables that are shared across processes
_model = None
_test_list_idices = None
_all_item_idices = None
_test_Remove = None

def evaluate_model(model, test_list_idices, all_item_idices, testRemove):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _test_list_idices
    global _all_item_idices
    global _test_Remove
    _model = model
    _test_list_idices = test_list_idices
    _all_item_idices = all_item_idices
    _test_Remove = testRemove

    i = 0
    # Single thread
    t1 = time()

    Final = {}
    # for idx in xrange(len(_test_list_idices)):
    #     i += 1
    #     if i % 10 == 0:
    #         t2 = time()
    #         print(i)
    #         print(t2-t1)
    #         t1 = time()
    #     Final[unicode(str(_test_list_idices[idx]), "utf-8")] = eval_one_list(idx)

    for u in _test_Remove.keys():
    # for u in [u'1003738']:
        i += 1
        if i % 10 == 0:
            t2 = time()
            print(i)
            print(t2-t1)
            t1 = time()
        Final[u] = eval_one_list(u)

    return Final


def eval_one_list(u):
    # u = _test_list_idices[idx]
    # u = _test_Remove.keys()[idx]
    print u
    print(len(_test_Remove[u]))
    print(_test_Remove[u])

    # items = range(_num_items)
    items = _all_item_idices


    # print(sum(np.array(items) == _test_Remove[u][0]))
    # print(sum(np.array(items) == _test_Remove[u][1]))
    # print(sum(np.array(items) == _test_Remove[u][2]))
    # print(sum(np.array(items) == _test_Remove[u][3]))
    # print(sum(np.array(items) == _test_Remove[u][4]))

    print(len(items))
    if _test_Remove[u] != []:
        tmp = set(_test_Remove[u])
        items1 = list(set(items) - tmp)
    else:
        items1 = items
        tmp = set([])
    print(len(items1))

    # print(sum(np.array(items1) == tmp[0]))
    # print(sum(np.array(items1) == tmp[1]))
    # print(sum(np.array(items1) == tmp[2]))
    # print(sum(np.array(items1) == tmp[3]))
    # print(sum(np.array(items1) == tmp[4]))

    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items1), u, dtype='int32')
    predictions = _model.predict([users, np.array(items1)],
                                 batch_size=100, verbose=0)
    print(len(predictions))
    for i in xrange(len(items1)):
        item = items1[i]
        map_item_score[item] = predictions[i]
    # items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(500, map_item_score, key=map_item_score.get)

    if len(set(ranklist)-tmp) != 500:
        print('Wrong!')

    return ranklist


