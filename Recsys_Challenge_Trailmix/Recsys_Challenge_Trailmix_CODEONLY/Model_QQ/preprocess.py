# import json
# import numpy as np
#
# a = json.load(open('PL_TRACKS_ALL.json'))
#
# x = 0  # '864737' 376
# y = 1000  # '54' 5
# for i in range(len(a)):
#     if len(a[str(i)]) > x:
#         x = len(a[str(i)])
#         g = i
#     if len(a[str(i)]) < y:
#         y = len(a[str(i)])
#         k = i
#
#
# name = []
# for i in range(len(a)):
#     #if i % 10000 == 0:
#     print(i)
#     for j in range(len(a[str(i)])):
#         if a[str(i)][j] not in name:
#             name.append(a[str(i)][j])
#
# for i in range(len(a)):
#     if i % 10000 == 0:
#         print(i)
#     for j in range(len(a[str(i)])):
#         name.append(a[str(i)][j])
#
# name = np.unique(name)
#
#
#
#
# b = np.asarray(name)
# # np.save('song_name.py', b)
# b = np.load('song_name.py.npy')
#
# import copy
# c = copy.deepcopy(a)
# num_of_song = len(b)
#
# # for i in range(len(a)):
# #     #if i % 10000 == 0:
# #     print(i)
# #     for j in range(len(a[str(i)])):
# #         c[str(i)][j] = int(np.where(b == a[str(i)][j])[0])
# #         # for k in range(num_of_song):
# #         #     if b[k] == a[str(i)][j]:
# #         #         c[str(i)][j] = k+1
#
# map = np.zeros([8, 75, 75], dtype=np.int)
# for i in range(len(b)):
#     map[ord(b[i][0])-48, ord(b[i][1]) - 48, ord(b[i][2]) - 48] += 1
#
#
# Cmap = np.reshape(np.cumsum(map), [8, 75, 75])
#
# for i in range(len(a)):
#     if i % 10000 == 0:
#         print(i)
#     for j in range(len(a[str(i)])):
#         tmp1 = ord(a[str(i)][j][0]) - 48
#         tmp2 = ord(a[str(i)][j][1]) - 48
#         tmp3 = ord(a[str(i)][j][2]) - 48
#         s2 = Cmap[tmp1, tmp2, tmp3]
#         s1 = Cmap[tmp1, tmp2, tmp3] - map[tmp1, tmp2, tmp3]
#         tmpb = b[s1:s2]
#         c[str(i)][j] = s1 + int(np.where(tmpb == a[str(i)][j])[0])
#
# qq = 0
# for i in range(len(a)):
#     if i % 10000 == 0:
#         print(i)
#     for j in range(len(a[str(i)])):
#         if a[str(i)][j] == b[c[str(i)][j]]:
#             qq += 1
#
#
# # with open('Map.json', 'w') as fp:
# #     json.dump(c, fp)
#
# c = json.load(open('Map.json'))
#
# output = open("all.txt", 'a')
# for i in range(len(a)):
#     if i % 10000 == 0:
#         print(i)
#     for j in range(len(a[str(i)])):
#         line = str(i) + '\t' + str(c[str(i)][j]) + '\n'
#         output.write(line)
# output.close()
#
# with open("all.txt") as in_f:
#     num_of_rating = 0
#     for line in in_f:
#         num_of_rating += 1
# print(num_of_rating) # 10000054
#
# i = 0
# qq = []
# for line in file("Data/all_NeuMF.txt"):
#     if i % 10000 == 0:
#         print(i)
#     i += 1
#     data = line.rstrip('\n').split('\t')
#     qq.append(int(data[1]))
#
# max(qq)
# print(line)
# print(data)
#
# # i_idx = str(0)
# # mark = str(0)
# # outfile = file("netflix/u_u_5.txt", "w")
# # for line in file("netflix/u_u_4.txt"):
# #     data = line.rstrip('\n').split('\t')
# #     if i_idx == data[1]:
# #         data.append(mark)
# #         mark = data[3]
# #     else:
# #         data.append(str(0))
# #         mark = data[3]
# #         i_idx = data[1]
# #     outfile.write('\t'.join(data))
# #     outfile.write('\n')
# # outfile.close()
#
#
# # for i in range(len(a)):
# #     #if i % 10000 == 0:
# #     print(i)
# #     for j in range(len(a[str(i)])):
# #         c[str(i)][j] = int(np.where(b == a[str(i)][j])[0])
# #         # for k in range(num_of_song):
# #         #     if b[k] == a[str(i)][j]:
# #         #         c[str(i)][j] = k+1
#
#
#
# # for i in range(len(a)):
# #     if i % 10000 == 0:
# #         print(i)
# #     for j in range(len(a[str(i)])):
# #         tmp1 = ord(a[str(i)][j][0]) - 48
# #         tmp2 = ord(a[str(i)][j][1]) - 48
# #         tmp3 = ord(a[str(i)][j][2]) - 48
# #         if tmp1 == 0 and tmp2 == 0 and tmp3 == 0:
# #             s1 = 0
# #         elif tmp1 == 0 and tmp2 == 0:
# #             s1 = sum(map[0, 0, 0:tmp3])
# #         elif tmp2 == 0 and tmp3 == 0:
# #             s1 = sum(sum(sum(map[0:tmp1, :, :])))
# #         elif tmp1 == 0 and tmp3 == 0:
# #             s1 = sum(sum(map[0, 0:tmp2, :]))
# #         elif tmp1 == 0:
# #             s1 = sum(sum(map[0, 0:tmp2, :])) + sum(map[0, tmp2, 0:tmp3])
# #         elif tmp2 == 0:
# #             s1 = sum(sum(sum(map[0:tmp1, :, :]))) + sum(map[tmp1, tmp2, 0:tmp3])
# #         elif tmp3 == 0:
# #             s1 = sum(sum(sum(map[0:tmp1, :, :]))) + sum(sum(map[tmp1, 0:tmp2, :]))
# #         else:
# #             s1 = sum(sum(sum(map[0:tmp1, :, :])))\
# #                  + sum(sum(map[tmp1, 0:tmp2, :])) + sum(map[tmp1, tmp2, 0:tmp3])
# #         s2 = s1 + map[tmp1, tmp2, tmp3]
# #         tmpb = b[s1:s2]
# #         c[str(i)][j] = s1 + int(np.where(tmpb == a[str(i)][j])[0])
#
# # qq1 = []
# # qq2 = []
# # for line in file("Data/all_NeuMF.txt"):
# #     data = line.rstrip('\n').split('\t')
# #     qq1.append(int(data[0]))
# #     qq2.append(int(data[1]))
#


# qq1 = []
# for line in file("Data/Task_2/PL.train.rating.txt"):
#     data = line.rstrip('\n').split('\t')
#     if int(data[0]) == 7:
#         print(data)
#         qq1.append(int(data[1]))
# print(qq1)


# import numpy as np
# b = np.load('song_name.py.npy')
#
# map = np.zeros([8, 75, 75], dtype=np.int)
# for i in range(len(b)):
#     map[ord(b[i][0])-48, ord(b[i][1]) - 48, ord(b[i][2]) - 48] += 1
#
#
# Cmap = np.reshape(np.cumsum(map), [8, 75, 75])
# np.save('Cmap', Cmap)
# np.save('map', map)



# import json
# import numpy as np
#
# #tmp = '5_TEST_T9'
# #filename = 'PL_TRACKS_'+tmp+'.json'
# filename = 'PL_TRACKS_ALL.json'
# a = json.load(open(filename))
# Cmap = np.load('Cmap.npy')
# map1 = np.load('map.npy')
# b = np.load('song_name.py.npy')
#
# import copy
# c = copy.deepcopy(a)
#
# k = 0
# for i in a.keys():
#     k += 1
#     if k % 10000 == 0:
#         print(k)
#     for j in range(len(a[i])):
#         tmp1 = ord(a[str(i)][j][0]) - 48
#         tmp2 = ord(a[str(i)][j][1]) - 48
#         tmp3 = ord(a[str(i)][j][2]) - 48
#         s2 = Cmap[tmp1, tmp2, tmp3]
#         s1 = Cmap[tmp1, tmp2, tmp3] - map1[tmp1, tmp2, tmp3]
#         tmpb = b[s1:s2]
#         c[i][j] = s1 + int(np.where(tmpb == a[i][j])[0])
#
# # ff = 'Data/'+tmp+'.json'
# ff = 'PL_TRACKS_ALL_MAP.json'
# with open(ff, 'w') as fp:
#     json.dump(c, fp)


# import json
# import numpy as np
#
# tmp = 'ALL'
# filename = 'PL_TRACKS_'+tmp+'.json'
# a = json.load(open(filename))
#
# qq = []
# k = 0
# for i in a.keys():
#     k += 1
#     if k % 10000 == 0:
#         print(k)
#     if len(a[i]) != len(set(a[i])):
#         print(i)
#         qq.append(i)
#
# print len(qq)


# Cmap = np.load('Cmap.npy')
# qq=0
# for i in xrange(8):
#     for j in xrange(75):
#         for k in xrange(75):
#             if i*j*k!=0:
#                 if Cmap[i,0,k]<Cmap[i-1,74,k]:
#                     qq+=1
#
#
#
#
# 053xKa7PdxQsJNWmBjV0sv
#
# 053xKa7PdxQsJNWmBjV0sv

# import json
# import numpy as np
#
# filename = 'Data/ALL.json'
# a = json.load(open(filename))
#
# t = np.zeros(2300000)
# np.save('track_stat.npy', t)
#
# k = 0
# for i in a.keys():
#     k += 1
#     if k % 10000 == 0:
#         print k
#     tmp = a[i]
#     a[i] = list(set(a[i]))
#     if len(a[i]) != 0:
#         for j in xrange(len(a[i])):
#             t[a[i][j]] += 1
#
# np.save('track_stat.npy', t)
















# import numpy as np
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
#
# t = np.load('track_stat.npy')
#
#
# g = t[t<100]
#
# plt.hist(t, 10)
#
#
# a20 = np.where(t>20)[0]
# a50 = np.where(t>50)[0]
# a100 = np.where(t>100)[0]
# a500 = np.where(t>500)[0]
#
# np.save('item_idx_20_up.npy', a20)
# np.save('item_idx_50_up.npy', a50)
# np.save('item_idx_100_up.npy', a100)
# np.save('item_idx_500_up.npy', a500)







# import numpy as np
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
#
# t = np.load('track_stat.npy')
#
# n = 100
# qq = np.zeros(n)
# for i in range(n):
#     qq[i] = len(t[(t>5*i)&(t<=5*(i+1))])/2262292.0
#
#
# qq2 = np.zeros(n)
# for i in range(n):
#     qq2[i] = sum(t[(t>5*i)&(t<=5*(i+1))])/sum(t)
#
# # plt.plot(qq, qq2)
# plt.figure(3)
# plt.plot(range(0, 500, 5), qq2[0:100])
#
#
# plt.plot(range(0, 100, 5), np.cumsum(qq2))
#
# plt.figure(2)
# plt.plot(range(0, 100, 5), np.cumsum(qq[0:20]))
#
#
# plt.figure(4)
# plt.plot(range(0, 500, 5), qq2*1.0/qq)




# import json
# import numpy as np
#
# filename = 'Data/ALL.json'
# a = json.load(open(filename))
#
# t = np.zeros([2262292, 350])
# np.save('track_stat_2.npy', t)
#
# k = 0
# for i in a.keys():
#     k += 1
#     if k % 10000 == 0:
#         print k
#     tmp = a[i]
#     a[i] = list(set(a[i]))
#     if len(a[i]) != 0:
#         for j in xrange(len(a[i])):
#             t[a[i][j], j] += 1
#
# np.save('track_stat_2.npy', t)








# import numpy as np
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
#
# t = np.load('track_stat_2.npy')
#
# # >1
# tmp = np.sum(t[:, 1:], 1)
# a0 = np.where(tmp>0)[0]
# a5 = np.where(tmp>5)[0]
# a20 = np.where(tmp>20)[0]
# a50 = np.where(tmp>50)[0]
# a100 = np.where(tmp>100)[0]
# a500 = np.where(tmp>500)[0]
# np.save('item_idx_0_up_1.npy', a0)
# np.save('item_idx_5_up_1.npy', a5)
# np.save('item_idx_20_up_1.npy', a20)
# np.save('item_idx_50_up_1.npy', a50)
# np.save('item_idx_100_up_1.npy', a100)
# np.save('item_idx_500_up_1.npy', a500)
#
#
# # >5
# tmp = np.sum(t[:, 5:], 1)
# a0 = np.where(tmp>0)[0]
# a5 = np.where(tmp>5)[0]
# a20 = np.where(tmp>20)[0]
# a50 = np.where(tmp>50)[0]
# a100 = np.where(tmp>100)[0]
# a500 = np.where(tmp>500)[0]
# np.save('item_idx_0_up_5.npy', a0)
# np.save('item_idx_5_up_5.npy', a5)
# np.save('item_idx_20_up_5.npy', a20)
# np.save('item_idx_50_up_5.npy', a50)
# np.save('item_idx_100_up_5.npy', a100)
# np.save('item_idx_500_up_5.npy', a500)
#
# # >25
# tmp = np.sum(t[:, 25:], 1)
# a0 = np.where(tmp>0)[0]
# a5 = np.where(tmp>5)[0]
# a20 = np.where(tmp>20)[0]
# a50 = np.where(tmp>50)[0]
# a100 = np.where(tmp>100)[0]
# a500 = np.where(tmp>500)[0]
# np.save('item_idx_0_up_25.npy', a0)
# np.save('item_idx_5_up_25.npy', a5)
# np.save('item_idx_20_up_25.npy', a20)
# np.save('item_idx_50_up_25.npy', a50)
# np.save('item_idx_100_up_25.npy', a100)
# np.save('item_idx_500_up_25.npy', a500)
#
# # >100
# tmp = np.sum(t[:, 100:], 1)
# a0 = np.where(tmp>0)[0]
# a5 = np.where(tmp>5)[0]
# a20 = np.where(tmp>20)[0]
# a50 = np.where(tmp>50)[0]
# a100 = np.where(tmp>100)[0]
# a500 = np.where(tmp>500)[0]
# np.save('item_idx_0_up_100.npy', a0)
# np.save('item_idx_5_up_100.npy', a5)
# np.save('item_idx_20_up_100.npy', a20)
# np.save('item_idx_50_up_100.npy', a50)
# np.save('item_idx_100_up_100.npy', a100)
# np.save('item_idx_500_up_100.npy', a500)

# import json
# import numpy as np
# q = json.load(open('PL_NUM_HOLDOUTS_READONLY.json'))
# ori = json.load(open('PL_NUM_TRACKS_READONLY.json'))
# t = np.array(q.values())
#
# t_ori = np.array(ori.values())
#
# diff = t_ori-t

# import json
# a = json.load(open('PL_TRACKS_ALL_MAP.json'))
#
# k = 0
# t = 0
# for i in a.keys():
#     k += 1
#     if k % 10000 == 0:
#         print(k)
#     t += len(list(set(a[i])))
# print(t)
#
#
#
# import json
# a = json.load(open('PL_TRACKS_FINAL_TEST.json'))
#
# k = 0
# t = 0
# for i in a.keys():
#     k += 1
#     if k % 10000 == 0:
#         print(k)
#     t += len(list(set(a[i])))
# print(t)



# a = json.load(open('PL_TRACKS_FINAL_TEST.json'))
# a['1003738']
# [754961, 1203346, 1974353, 1040207, 1498381]
# a = json.load(open('WAR_PL_TRACKS_READONLY.json'))
# a['1003738']
# [u'2aibwv5hGXSgw7Yru8IYTO', u'48UPSzbZjgc449aqz8bxox', u'6nTiIhLmQ3FWhvrGafw2zj', u'3ZffCQKLFLUvYM59XKLbVm', u'59WN2psjkt1tyaxjspN8fp']
# b = np.load('song_name.py.npy')
# Traceback (most recent call last):
#   File "<input>", line 1, in <module>
# NameError: name 'np' is not defined
# import numpy as np
# b = np.load('song_name.py.npy')




# import json
# import numpy as np
# #tmp = '5_TEST_T7'
# #filename = 'PL_TRACKS_'+tmp+'.json'
# #filename = 'Data/TryT5.json'
# filename = 'QQ_submit_word2vec.json'
# a = json.load(open(filename))
# Cmap = np.load('Cmap.npy')
# map1 = np.load('map.npy')
# b = np.load('song_name.py.npy')
#
# import copy
# c = copy.deepcopy(a)
#
# k = 0
# for i in a.keys():
#     k += 1
#     if k % 10000 == 0:
#         print(k)
#     for j in range(len(a[i])):
#         tmp1 = ord(a[str(i)][j][0]) - 48
#         tmp2 = ord(a[str(i)][j][1]) - 48
#         tmp3 = ord(a[str(i)][j][2]) - 48
#         s2 = Cmap[tmp1, tmp2, tmp3]
#         s1 = Cmap[tmp1, tmp2, tmp3] - map1[tmp1, tmp2, tmp3]
#         tmpb = b[s1:s2]
#         c[i][j] = s1 + int(np.where(tmpb == a[i][j])[0])
#
# # ff = 'Data/'+tmp+'.json'
# ff = 'QQ_submit_word2vec_MAP.json'
# # ff = 'Data/Try.json'
# with open(ff, 'w') as fp:
#     json.dump(c, fp)


import json
a = json.load(open('xing.json'))
b = json.load(open('QQ_submit_word2vec_MAP.json'))
c = json.load(open('PL_NUM_HOLDOUTS_READONLY.json'))
d = json.load(open('PL_NUM_TRACKS_READONLY.json'))

qq = {}
xing = {}

for i in b.keys():
    qq[i] = b[i][0:c[i]]

for i in a.keys():
    xing[i] = a[i][0:c[i]]


import numpy as np
num = np.zeros(10000)

k = 0
for i in qq.keys():
    num[k] = len(set(qq[i]).intersection(set(xing[i])))
    k += 1

total = 0
for i in qq.keys():
    total += len(qq[i])


ori = np.zeros(10000)

k = 0
for i in qq.keys():
    ori[k] = len(set(qq[i]))
    k+=1


gg = num/ori
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
plt.plot(gg)
plt.hist(gg,bins=50,normed=True)

 # Statistics are only based on 9000 users T1-T9 no T0
import numpy as np
idx = np.load('item_idx_100_up.npy')
qq_t = 0 # 670562
for i in qq.keys():
    for j in range(len(qq[i])):
        if qq[i][j] in idx:
            qq_t += 1

c_t = 0  # 670562
for i in qq.keys():
    for j in range(len(qq[i])):
        c_t += 1

xing_t = 0  # 657957 total: 670562 diff: 12605
for i in qq.keys():
    for j in range(len(qq[i])):
        if xing[i][j] in idx:
            xing_t += 1

xing_a_t = 0  # 4356849  total: 4500000  diff: 143151
for i in qq.keys():
    for j in range(len(a[i])):
        if a[i][j] in idx:
            xing_a_t += 1

qq_b_t = 0  # 4500000
for i in qq.keys():
    for j in range(len(b[i])):
        if b[i][j] in idx:
            qq_b_t += 1



# Overlap
k = 0
num = np.zeros(9000)
for i in qq.keys():
    num[k] = len(set(qq[i]).intersection(set(xing[i])))
    k += 1

k = 0
num2 = np.zeros(9000)
for i in qq.keys():
    num2[k] = len(set(b[i]).intersection(set(a[i])))
    k += 1


 # First 10/20/30 Tracks Statistics


N = np.zeros(9000) # 5-225, >10: 8891 >20: 8435 >30: 7948  >40: 6971 >50: 6119
k = 0
for i in qq.keys():
    N[k] = c[i]
    k += 1


# qq top 10/20/30 in Xing


# top 5
N5 = np.zeros(9000)
k = 0
for i in qq.keys():
    for j in range(5):
        if b[i][j] in xing[i]:
            N5[k] += 1
    k += 1


# top 10
N10 = np.zeros(9000)
k = 0
for i in qq.keys():
    for j in range(min(10, c[i])):
        if b[i][j] in xing[i][0:min(10, c[i])]:
            N10[k] += 1
    k += 1

Ng10 = np.minimum(N10, N)



N20 = np.zeros(9000)
k = 0
for i in qq.keys():
    for j in range(20):
        if b[i][j] in xing[i]:
            N20[k] += 1
    k += 1

Ng20 = np.minimum(N20, N)



N30 = np.zeros(9000)
k = 0
for i in qq.keys():
    for j in range(30):
        if b[i][j] in xing[i]:
            N30[k] += 1
    k += 1

Ng30 = np.minimum(N30, N)

                        #
# # Change to orginal
# import numpy as np
# import json
# filename = 'Data/TryT5.json'
# Final = json.load(open(filename))
# b = np.load('song_name.py.npy')
# k = 0
# for p in Final.keys():
#     k += 1
#     if k % 10000 == 0:
#         print(k)
#     for q in range(len(Final[p])):
#         Final[p][q] = b[Final[p][q]]
#
#
# ff = 'Data/TryT5Final.json'
# with open(ff, 'w') as fp:
#      json.dump(Final, fp)


# import json
#
# a = json.load(open('PL_TRACKS_ALL_MAP.json'))
# for i in a.keys():
#     a[i] = [str(x) for x in a[i]]
#
# ff = 'PL_TRACKS_ALL_MAP_STR.json'
# with open(ff, 'w') as fp:
#      json.dump(a, fp)







# import json
# import numpy as np


# Final = json.load(open('Data/TryT5.json'))
# b = np.load('song_name.py.npy')
# k = 0
# for p in Final.keys():
#     k += 1
#     if k % 10000 == 0:
#         print(k)
#     for q in range(len(Final[p])):
#         Final[p][q] = b[Final[p][q]]
#
# ff = 'Data/T5.json'
# with open(ff, 'w') as fp:
#     json.dump(Final, fp)
#
#
#
# Final = json.load(open('Data/TryT2.json'))
# b = np.load('song_name.py.npy')
# k = 0
# for p in Final.keys():
#     k += 1
#     if k % 10000 == 0:
#         print(k)
#     for q in range(len(Final[p])):
#         Final[p][q] = b[Final[p][q]]
#
# ff = 'Data/T2.json'
# with open(ff, 'w') as fp:
#     json.dump(Final, fp)
#
#
# Final = json.load(open('Data/TryT6.json'))
# b = np.load('song_name.py.npy')
# k = 0
# for p in Final.keys():
#     k += 1
#     if k % 10000 == 0:
#         print(k)
#     for q in range(len(Final[p])):
#         Final[p][q] = b[Final[p][q]]
#
# ff = 'Data/T6.json'
# with open(ff, 'w') as fp:
#     json.dump(Final, fp)



import json
import numpy as np

a = json.load(open('PL_TRACKS_FINAL_TEST.json'))
# t = np.load('item_idx_100_up.npy')
t = np.load('item_idx_5_up.npy')


t1 = np.zeros(10000)
t2 = np.zeros(10000)
k = 0
for i in a.keys():
    t1[k] = len(a[i])
    tmp = 0
    for j in a[i]:
        if j in t:
            tmp+=1
    t2[k] = tmp
    k += 1
