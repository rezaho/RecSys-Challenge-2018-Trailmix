import json
import random
import numpy
import itertools
from collections import Counter
import operator
import numpy as np
import math
import sys
import gensim

start = int(sys.argv[1]) * 1250

print ('Start Now 0618!!~!')

def recommador(related_songs, already_have, num = 500):
    sorted_song = sorted(related_songs.items(), key=operator.itemgetter(1), reverse=True)
    result = []
    for pair in sorted_song:
        if not pair[0] in already_have:
            result.append(pair[0])
        if len(result) == num:
            break
    return result


track_index = json.load(open('../DATA_PROCESSING/ALL_INDEX_READONLY/TRACK_INDEX_READONLY.json'))

train_raw = json.load(open('../DATA_PROCESSING/PL_TRACKS_ALL.json'))
train = {}
for pl in train_raw:
    train[pl] = []
    for t in train_raw[pl]:
        train[pl].append(track_index[t])

del train_raw

track_artist_album_indexed = json.load(open('../DATA_PROCESSING/ALL_INDEX_READONLY/TRACK_ARTIST_ALBUM_INDEXED.json'))

train_artist_indexed = json.load(open('../DATA_PROCESSING/ALL_INDEX_READONLY/PL_ARTISTS_ALL_INDEXED.json'))
train_album_indexed = json.load(open('../DATA_PROCESSING/ALL_INDEX_READONLY/PL_ALBUMS_ALL_INDEXED.json'))

WAR_raw = json.load(open('../DATA_PROCESSING/WAR_PL_TRACKS_READONLY.json'))

WAR = {}
for pl in WAR_raw:
    #print pl
    WAR[pl] = []
    for t in WAR_raw[pl]:
        WAR[pl].append(track_index[t])
        
del WAR_raw

small_test_war = {}
for elem in list(WAR.keys())[start : start + 1250]:
    small_test_war[elem] = WAR[elem]

small_test_war_artist = {}
for pl in small_test_war:
    small_test_war_artist[pl] = []
    for t in small_test_war[pl]:
        small_test_war_artist[pl].append(track_artist_album_indexed[str(t)]['artist'])
        
small_test_war_album = {}
for pl in small_test_war:
    small_test_war_album[pl] = []
    for t in small_test_war[pl]:
        small_test_war_album[pl].append(track_artist_album_indexed[str(t)]['album'])

Most_Popular_2000 = []
with open('../MODEL_0_MOST_POPULAR/POPULARITY_RANKED_TRACK_2000.txt') as f:
    for line in f:
        Most_Popular_2000.append(track_index[line.replace('\n','')])

R = {}
i = start

for pl_test in small_test_war.keys():
    if i % 10 == 0:
        print (i)
    i += 1
    
    current = {}
    this = small_test_war[pl_test]
    this_artist = small_test_war_artist[pl_test]
    this_album = small_test_war_album[pl_test]
    
    if len(this) == 0:
        R[pl_test] = []
        continue
    
    elif len(this) < 25:
        bound = 1
    elif len(this) < 100:
        bound = 5
    elif len(this) == 100:
        bound = 10
        
    for pl_train in train:
        that = train[pl_train]
        that_artist = train_artist_indexed[pl_train]
        that_album = train_album_indexed[pl_train]
        
        common_track = list(set(this) & set(that))
        common_artist = list(set(this_artist) & set(that_artist))
        common_album = list(set(this_album) & set(that_album))
        
        #all_common_len = common_track + conmon_len_artist * 0.3 + conmon_len_album * 0.6
        
        if len(common_track) >= bound:
            for t in that:
                if track_artist_album_indexed[str(t)]['artist'] in common_artist:
                    common_len_artist = len(common_artist)
                else:
                    common_len_artist = 0
                    
                if track_artist_album_indexed[str(t)]['album'] in common_album:
                    common_len_album = len(common_album)
                else:
                    common_len_album = 0
                
                
                all_common_len = len(common_track) + common_len_artist * 0.1 + common_len_album * 0.2
                    
                    
                try:
                    #current[t] += math.pow(common_len, 3) / math.pow(ALL_TRACK_FREQUENCE[track_index_reverse[str(t)]], 1.0/3)
                    current[t] += math.pow(all_common_len, 3)
                except:
                    current[t] = math.pow(all_common_len, 3)
                    #current[t] = math.pow(common_len, 3)/ math.pow(ALL_TRACK_FREQUENCE[track_index_reverse[str(t)]], 1.0/3)
                    
    temp = recommador(current, small_test_war[pl_test])

    R[pl_test] = temp

# Split to Two Model Tasks
R_Model_5 = {}
for pl in R:
    if len(R[pl]) == 500:
        R_Model_5[pl] = R[pl][:]

R_Model_2 = {}
for pl in R:
    if len(R[pl]) < 500:
        R_Model_2[pl] = R[pl][:]

del R

model = gensim.models.Word2Vec.load('../MODEL_2_JL_WORD2VEC_PLSIM/Song2Vec_For_1M_Challenge/Song2Vec_for_1M_Challenge')
track_index_reversed = {k:v for (v, k) in track_index.items()}

def find_nearest_k(pid, already_have = [], k = 500):
    global small_test_war
    
    res = []
    
    this_positive = []
    
    for elem in small_test_war[pl]:
        if track_index_reversed[elem] in model.wv.vocab:
            this_positive.append(track_index_reversed[elem])
    
    if this_positive != []:
        top1000_pair = model.wv.most_similar(positive = this_positive, topn=1000)
    
        for pair in top1000_pair:
            if not (track_index[pair[0]] in small_test_war[pid]) and (not track_index[pair[0]] in already_have):
                res.append(track_index[pair[0]])
            if len(res) == k:
                break
        
    else:
        for t_index in Most_Popular_2000:
            if not (t_index in small_test_war[pid]) and (not t_index in already_have):
                res.append(t_index)
            if len(res) == k:
                break
                
    return res

R_Model_2_pure = {}
for pl in R_Model_2:
    current = find_nearest_k(pl, already_have = R_Model_2[pl], k = 500 - len(R_Model_2[pl]))

    R_Model_2_pure[pl] = current

for pl in R_Model_2:
    R_Model_2[pl] += R_Model_2_pure[pl][:]

#R_new = dict(R_Model_2.items() + R_Model_5.items())
R_new = {**R_Model_2, **R_Model_5}

#final check

for pl in R_new:
    if len(R_new[pl]) == 0:
        R_new[pl] = Most_Popular_2000[:500]
        print ("Some Thing Wrong")

for pl in R_new:
    if len(list(set(R_new[pl]))) != 500:
        print ("Some Thing Wrong: Not Unique")

# index to track id
R_id = {}
for pl in R_new:
    R_id[pl] = []
    if len(R_new[pl]) != 0:
        for ti in R_new[pl]:
            R_id[pl].append(track_index_reversed[ti])

with open('../WAR/0618/WAR_PREDICT_0618/Submission_0618_WAR_P_' + sys.argv[1] + '.json', 'w') as f:
	json.dump(R_id, f)


