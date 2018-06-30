'''
Created on Aug 8, 2016
Processing datasets.
@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import json

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.test_list_idices, self.testRemove \
            = self.load_rating_file_as_list('PL_TRACKS_FINAL_TEST.json')
        # self.trainMatrix, self.trainRatings, self.all_item_idices \
        #     = self.load_rating_file_as_matrix('PL_TRACKS_ALL_MAP.json', 'PL_TRACKS_FINAL_TEST.json')
        #
        # self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename2):

        Test_Remove = json.load(open(filename2))
        users = []

        k = 0
        for i in Test_Remove.keys():
            k += 1
            if k % 10000 == 0:
                print(k)
            users.append(int(i))


        users = list(set(users))
        users.sort()

        return users, Test_Remove


    def load_rating_file_as_matrix(self, filename1, filename2):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''

        # Add Back Test_Remove to Original Train_ALL
        Train_ALL = json.load(open(filename1)) # 'PL_TRACKS_5_ALL.json
        Test_Remove = json.load(open(filename2))

        k = 0
        for i in Test_Remove.keys():
            k += 1
            if k % 10000 == 0:
                print(k)
            Train_ALL[i] = Test_Remove[i]

        # Get number of users and items
        num_users = int(max(Test_Remove.keys()))




        all_item_idices = []
        k = 0
        for i in Train_ALL.keys():
            k += 1
            if k % 10000 == 0:
                print(k)
            all_item_idices += Train_ALL[i]

        all_item_idices = list(set(all_item_idices))
        all_item_idices.sort()
        num_items = max(all_item_idices)

        print('qqqqqqqqq')
        print(num_users)
        print('qqqqqqqqq')


        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)

        k = 0
        for i in Train_ALL.keys():
            k += 1
            if k % 10000 == 0:
                print(k)
            # for j in xrange(len(Train_ALL[i])):
            mat[int(i), Train_ALL[i]] = 1.0

        # Construct List
        ratings = []
        k = 0
        for i in Train_ALL.keys():
            k += 1
            if k % 10000 == 0:
                print(k)
            rating_list = list(set(Train_ALL[i]))
            ratings.append(rating_list)

        return mat, ratings, all_item_idices