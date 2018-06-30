import gensim
import numpy as np
import json

model = gensim.models.Word2Vec.load('xing_song2vec/song2vec_QQ_1M_MAP') # MODEL_2_JL_WORD2VEC_PLSIM/song2vec_GT5_TRAIN
# model = gensim.models.Word2Vec.load('song2vec_QQ_1M_MAP') # MODEL_2_JL_WORD2VEC_PLSIM/song2vec_GT5_TRAIN
print len(model.wv.vocab)
# Cmap = np.load('Cmap.npy')
# map1 = np.load('map.npy')
# b = np.load('song_name.py.npy')
k = 10000

# playlist = json.load(open('PL_TRACKS_ALL_MAP.json'))
# Final = {}
# t = 0
# # for i in model.wv.vocab.keys():
# for i in playlist.keys():
#     t += 1
#     if t % 100 == 0:
#         print t
#     if playlist[i] != []:
#         p2str = [str(x) for x in playlist[i]]
#         tmp = model.most_similar(positive=p2str, topn=k)
#         mylist = []
#         for j in range(k):
#             # tmp0 = tmp[j][0]
#             # tmp1 = ord(tmp0[0]) - 48
#             # tmp2 = ord(tmp0[1]) - 48
#             # tmp3 = ord(tmp0[2]) - 48
#             # s2 = Cmap[tmp1, tmp2, tmp3]
#             # s1 = Cmap[tmp1, tmp2, tmp3] - map1[tmp1, tmp2, tmp3]
#             # tmpb = b[s1:s2]
#             # mylist.append(s1 + int(np.where(tmpb == tmp0)[0]))
#             mylist.append(int(tmp[j][0]))
#         Final[i] = mylist
#
#
# ff = 'word2vec_100W_train.json'
# with open(ff, 'w') as fp:
#     json.dump(Final, fp)



# for q in range(10):
#     t = 0
#     Final = {}
#     myfile = 'Keys/1000_Key_TEST_T'+str(q)+'.json'  # 'Data/TEST_T'+str(q)+'.json'
#     playlist_key = json.load(open(myfile))
#     myfile2 = 'Data/TEST_T'+str(q)+'.json'
#     playlist_test = json.load(open(myfile2))
#     for i in playlist_key.keys():
#         t += 1
#         if t % 100 == 0:
#             print t
#         if playlist_test[i] != []:
#             p2str = [str(x) for x in playlist_test[i]]
#             tmp = model.most_similar(positive=p2str, topn=k)
#             mylist = []
#             for j in range(k):
#                 mylist.append(int(tmp[j][0]))
#             Final[i] = mylist
#
#     ff = 'word2vec_similar_playlist/T'+str(q)+'.json'
#     with open(ff, 'w') as fp:
#         json.dump(Final, fp)
#





# t = 0
# Final = {}
# playlist_test = json.load(open('PL_TRACKS_FINAL_TEST.json'))
# for i in playlist_test.keys():
#     t += 1
#     if t % 100 == 0:
#         print t
#     if playlist_test[i] != []:
#         p2str = [str(x) for x in playlist_test[i]]
#         tmp = model.most_similar(positive=p2str, topn=k)
#         mylist = []
#         for j in range(k):
#             mylist.append(int(tmp[j][0]))
#         Final[i] = mylist
#
# ff = 'word2vec_1W_test.json'
# with open(ff, 'w') as fp:
#     json.dump(Final, fp)




# t = 0
# Final = {}
# playlist_test = json.load(open('Data/TEST_T5.json'))
# # for i in model.wv.vocab.keys():
# for i in playlist_test.keys()[3000:3200]:
#     t += 1
#     if t % 100 == 0:
#         print t
#     mylist = []
#     for j in playlist_test[i]:
#         tmp = model.most_similar(positive=str(j), topn=k)
#         for gg in range(k):
#             mylist.append(int(tmp[gg][0]))
#         Final[i] = list(set(mylist))
#
# ff = 'word2vec_TryT5_3000_3200.json'
# with open(ff, 'w') as fp:
#     json.dump(Final, fp)

# t = 0
# Final = {}
# # for i in range(1600001, 2262292):
# for i in range(2200001, 2262292):
#     t += 1
#     if t % 100 == 0:
#         print t
#     tmp = model.most_similar(positive=str(i), topn=k)
#     Final[i] = [int(x[0]) for x in tmp]
#
#     # if i % 100000 == 0:
#     #     ff = 'word2vec_all_song_1000_similar'+str(i/100000)+'.json'
#     #     with open(ff, 'w') as fp:
#     #         json.dump(Final, fp)
#     #     Final = {}
#
# ff = 'word2vec_all_song_1000_similar23.json'
# with open(ff, 'w') as fp:
#     json.dump(Final, fp)


# a = 'word2vec_similar_song/word2vec_all_song_500.json'
# b = json.load(open(a))
# print('READ_FINISH!')
#
# Final = {}
# for i in b.keys():
#     Final[i] = b[i][0:100]
#
# print('PROCESS_FINISH!')
#
#
# ff = 'word2vec_similar_song/word2vec_all_song_100.json'
# with open(ff, 'w') as fp:
#     json.dump(Final, fp)
# print('SAVE_FINISH!')