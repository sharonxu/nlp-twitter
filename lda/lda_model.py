#!/usr/bin/env python

from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
import math
import numpy as np
import cPickle as pickle
import time

NUM = 83561

def split_corpus(corpus, n_docs, train_split = 0.9):

	"""
	Args: 
	corpus = list of documents
	n_docs = number of documents
	train_split = percent of docs used for training data

	Returns: training corpus, testing corpus
	"""
	np.random.seed(NUM)
	perm = np.random.permutation(n_docs)
	train_len = math.floor(train_split * n_docs)
	train_index = perm[:train_len]

	train_corpus = []
	test_corpus = []
	mask = np.zeros(n_docs)
	mask[train_index] = 1

	current_index = 0
	for doc in corpus:
		if mask[current_index]:
			train_corpus.append(doc)
		else:
			test_corpus.append(doc)
		current_index += 1

	return (train_corpus, test_corpus)

def fit_numtopics(train_corpus, test_corpus, id2word, num_topics_list, iters, workers, chunksize, logfilename, save=True):

	"""
	Args: 
	num_topics_list = list of number of topics, a model will be fitted for each
	save: indicates whether model should be saved
	Returns: topics_dict = a dictionary of topics lists, where the key is the number of topics
	"""
	topics_dict = {}
	logfile = open(logfilename, 'w')
	for num_topics in num_topics_list:
		
		print('training', num_topics)
		np.random.seed(NUM)

		start_time = time.time()
		model = LdaMulticore(corpus=train_corpus, id2word=id2word,
							 num_topics=num_topics, iterations=iters,
							 eval_every=None, workers=workers,
							 chunksize=chunksize)
		end_time = time.time()

		if save:
			fname = 'data\\orig_' + str(num_topics) + 'topics.lda'
			model.save(fname)

		per_word_bound = model.log_perplexity(test_corpus)
		perplexity = np.exp2(-1.0 * per_word_bound)

		logfile.write('\n' + 'num_topics: ' + str(num_topics) + '\n')
		logfile.write('perplexity: ' + str(perplexity) + '\n')
		logfile.write('train_time: ' + str(end_time - start_time) + '\n' + 'Topics: \n')

		topics = model.show_topics(num_topics=num_topics, num_words=20)
		topics_dict[str(num_topics)] = topics
		for topic in topics:
			logfile.write('\n\t' + topic.encode('ascii', 'ignore')  + '\n')

	logfile.close()		
	return topics_dict

def main():

	params = {
		'corpus_file': 'data\\origtweets_dtm.pkl',
		'dict_file': 'data\\origtweets_dict.pkl',
		'logfilename': 'data\\numtopics_results.txt',
		'train_split': 0.9,
		'iters': 500,
		'workers': 8,
		'chunksize': 1000
	}

	corpus_file = params['corpus_file']
	num_topics_list = [5, 10, 15, 20]
 
	id2word = corpora.Dictionary.load(params['dict_file'])

	with open(corpus_file) as pkl_file:
		corpus = pickle.load(pkl_file)

	train_corpus, test_corpus = split_corpus(corpus, len(corpus), params['train_split'])

	# fit all the models and save results to disk
	topics = fit_numtopics(
		train_corpus=train_corpus, 
		test_corpus=test_corpus,
		id2word=id2word,
		num_topics_list=num_topics_list,
		iters=params['iters'],
		workers=params['workers'],
		chunksize=params['chunksize'],
		logfilename = params['logfilename']
	)

	#Ex: to import saved model
	#import gensim
	#model = gensim.models.LdaModel.load('5topics.lda')


if __name__ == '__main__':
	main()
