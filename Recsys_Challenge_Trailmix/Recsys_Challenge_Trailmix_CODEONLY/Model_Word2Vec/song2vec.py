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
                #for pl in self.list_list[line]:
                    temp = line
		    #result = [[p1, p2] for p1 in temp for p2 in temp if p1 != p2]
                    #for i in result:
		    for i in range(len(temp)):
			shuffle(temp)
                        yield temp

if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	pls = []
	with open('PL_TRACKS_5_TRAIN.json') as fp:
    		for line in fp:
        		l = json.loads(line)
        		for cid in l:
            			pls.append( l[cid] )
	sentences = MySentences(pls) # a memory-friendly iterator
	model = gensim.models.Word2Vec(sentences,min_count=3,size=100, window=5, workers=8)
	model.save('song2vec_GT5_TRAIN')
