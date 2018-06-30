import gensim, logging
from random import shuffle
import json
import sys

class MySentences(object):
    def __init__(self, list_list):
        self.list_list = list_list
	#self.train = train
 
    def __iter__(self):
        for line in self.list_list:
            temp = line

            for i in range(len(temp)):
                shuffle(temp)
                yield temp

if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	pls = []
	with open('../../DATA_PROCESSING/ALL_1M_CHALLENGE_PL_TRACKS.json') as fp:
    		for line in fp:
        		l = json.loads(line)
        		for cid in l:
            			pls.append( l[cid] )
	sentences = MySentences(pls) # a memory-friendly iterator
	model = gensim.models.Word2Vec(sentences,min_count=3,size=100, window=5, workers=12)
	model.save('Song2Vec_for_1M_Challenge')
