import sklearn
import gensim
import numpy as np
import operator
import json
from sklearn.metrics.pairwise import cosine_similarity
import sys

start = int(sys.argv[1]) * 1500

print 'Start! ', sys.argv[1]

def find_nearest_song_by_song(pid, given_song_vec, k):
    a_T = np.reshape(given_song_vec, (1, 100))
    Z = cosine_similarity(a_T,model.wv.vectors)
    A = list(Z[0])
    B = sorted(range(len(A)), key=lambda k: A[k], reverse=True)
    C = [str(model.wv.index2word[i]) for i in B]
    
    rec = []
    for elem in C:
        if elem not in WAR[pid]:
            rec.append(elem)
        if len(rec) == k:
            break

    return rec

def find_nearest_song_by_pl(pid, k):
    current_pl_emb = np.zeros((100,))
    for t in WAR[pid]:
        current_pl_emb += model.wv[t]
    current_pl_emb = current_pl_emb * 1.0 / len(WAR[pid])
    return find_nearest_song_by_song(pid, current_pl_emb, k)

model = gensim.models.Word2Vec.load('Song2Vec_For_1M_Challenge/Song2Vec_for_1M_Challenge')

QQ_Index = json.load(open('/home/xing/notebooks/2018/Recsys2018_NEW/WAR/0625/WAR_PREDICT_0625/QQ_submit_word2vec_double2_100_500_50.json'))

WAR_raw = json.load(open('/home/xing/notebooks/2018/Recsys2018_NEW/DATA_PROCESSING/WAR_PL_TRACKS_READONLY.json'))

WAR = {}
for elem in QQ_Index.keys()[start : start + 1500]:
    #print elem
    WAR[elem] = WAR_raw[elem][:]
    
del WAR_raw

Pure_PLSIM_Word2Vec = {}
count = start
for pl in WAR:
    if count % 10 == 0:
        print count
    count += 1
    Pure_PLSIM_Word2Vec[pl] = find_nearest_song_by_pl(pl, 500)

with open('Pure_PLSIM_Word2Vec/Pure_PLSIM_Word2Vec_Final_Challenge_Part_' + str(sys.argv[1]) + '.json', 'w') as f:
    json.dump(Pure_PLSIM_Word2Vec, f)