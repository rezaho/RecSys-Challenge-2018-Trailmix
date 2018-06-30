import json
import random
import numpy
import itertools
from collections import Counter
import operator
import numpy as np
import math
import sys

start = int(sys.argv[1]) * 725

print ('Start Now 0621')

def recommador(related_songs, already_have, num = 500):
    sorted_song = sorted(related_songs.items(), key=operator.itemgetter(1), reverse=True)
    result = []
    for pair in sorted_song:
        if not pair[0] in already_have:
            result.append(pair[0])
        if len(result) == num:
            break
    return result

#track_index = json.load(open('../DATA_PROCESSING/ALL_INDEX_READONLY/TRACK_INDEX_READONLY.json'))

train = json.load(open('../DATA_PROCESSING/ALL_INDEX_READONLY/PL_TRACKS_ALL_5_INDEXED.json'))

track_artist_album_indexed = json.load(open('../DATA_PROCESSING/ALL_INDEX_READONLY/TRACK_ARTIST_ALBUM_INDEXED.json'))

train_artist_indexed_counter = json.load(open('../DATA_PROCESSING/ALL_INDEX_READONLY/PL_ARTISTS_ALL_INDEXED_COUNTER.json'))
train_album_indexed_counter = json.load(open('../DATA_PROCESSING/ALL_INDEX_READONLY/PL_ALBUMS_ALL_INDEXED_COUNTER.json'))

WAR_raw = json.load(open('../DATA_PROCESSING/ALL_INDEX_READONLY/WAR_PL_TRACKS_INDEXED_READONLY.json'))

small_test_war = {}
for pl in list(WAR_raw.keys())[start : start + 725]:
    small_test_war[pl] = WAR_raw[pl]

del WAR_raw

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

#QQ = json.load(open('QQ_submit_word2vec.json'))

R = {}
i = start

for pl_test in small_test_war.keys():
    if i % 10 == 0:
        print (i)
    i += 1
    
    current = {}
    this = small_test_war[pl_test]
    this_artist = dict(Counter(small_test_war_artist[pl_test]))
    this_album = dict(Counter(small_test_war_album[pl_test]))
    
    if len(this) == 0:
        R[pl_test] = []
        continue
        
    elif len(this) == 1:
        track_bound = 1
        album_bound = 2
        artist_bound = 2
        
    elif len(this) == 5:
        track_bound = 2
        album_bound = 4
        artist_bound = 4
    
    elif len(this) == 10:
        track_bound = 3
        album_bound = 6
        artist_bound = 8
        
    elif len(this) == 25:
        track_bound = 5
        album_bound = 9
        artist_bound = 13
        
    elif len(this) == 100:
        track_bound = 15
        album_bound = 25
        artist_bound = 25
        
    for pl_train in train:
        that = train[pl_train]
        that_artist = train_artist_indexed_counter[pl_train]
        that_album = train_album_indexed_counter[pl_train]
        
        
        common_track = list(set(this) & set(that))
        common_artist = list(set(this_artist.keys()) & set(that_artist.keys()))
        common_album = list(set(this_album.keys()) & set(that_album.keys()))
        
        commom_artist_counter = {}
        commom_album_counter = {}
        for a in common_artist:
            commom_artist_counter[a] = min(that_artist[a], this_artist[a])
            
        for a in common_album:
            commom_album_counter[a] = min(that_album[a], this_album[a])
        

        if len(common_track) >= track_bound or \
        sum(commom_artist_counter.values()) >= artist_bound or \
        sum(commom_album_counter.values()) >= album_bound:
            
            for t in that:
                
                if str(track_artist_album_indexed[str(t)]['artist']) in commom_artist_counter:
                    #print track_artist_album_indexed[str(t)]['artist']
                    common_len_artist = commom_artist_counter[str(track_artist_album_indexed[str(t)]['artist'])]
                else:
                    common_len_artist = 0
                    
                if str(track_artist_album_indexed[str(t)]['album']) in commom_album_counter:
                    common_len_album = commom_album_counter[str(track_artist_album_indexed[str(t)]['album'])]
                else:
                    common_len_album = 0

                    
                all_common_len = len(common_track) + common_len_artist * 0.2 + common_len_album * 0.4
                
                #print len(common_track), common_len_artist,common_len_album
                    
                    
                try:
                    #current[t] += math.pow(common_len, 3) / math.pow(ALL_TRACK_FREQUENCE[track_index_reverse[str(t)]], 1.0/3)
                    current[t] += math.pow(all_common_len, 3)
                except:
                    current[t] = math.pow(all_common_len, 3)
                    #current[t] = math.pow(common_len, 3)/ math.pow(ALL_TRACK_FREQUENCE[track_index_reverse[str(t)]], 1.0/3)
                    
    temp = recommador(current, small_test_war[pl_test])

    R[pl_test] = temp

# track_index_reversed = {k:v for (v, k) in track_index.items()}

# R_combined = {}
# for pl in R:
#     if len(R[pl]) == 500:
#         R_combined[pl] = [track_index_reversed[i] for i in R[pl]]
#     else:
#         rest = 500 - len(R[pl])
#         R_combined[pl] = [track_index_reversed[i] for i in R[pl]]
#     	for t in QQ[pl]:
#     		if t not in R_combined[pl]:
#     			R_combined[pl].append(t)

#     		if len(R_combined[pl]) == 500:
#     			break

# for pl in R_combined:
# 	if len(R_combined[pl]) != 500:
# 		print "something is wrong!"

with open('../WAR/0621/WAR_PREDICT_0621/Submission_0621_WAR_P_' + sys.argv[1] + '_NOT_COMBINED.json', 'w') as f:
	json.dump(R, f)