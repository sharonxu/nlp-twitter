#!/usr/bin/env python

import logging
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
import numpy as np
import pandas as pd
import cPickle as pickle
import time

def to_dense(sparse_vector, length):
    # convert vector format from sparse to dense
    # e.g., sparse_vector= [(0, 0.9), (3, 0.1)]
    # length =  5
    # dense_vector = [0.9, 0, 0, 0.1, 0]

    dense_vector = [0] * length
    for item in sparse_vector:
        dense_vector[item[0]] = item[1]

    return dense_vector

def main():
    options = {
        'corpus_file': 'data\\origtweets_dtm.pkl',
        'id_file': 'data\\row_origtweets.csv',
        'model_file': 'data\\orig_10topics.lda',
        'meta_file': 'data\\origtweets_meta.csv',
        'output_file': 'data\\origtweets_topics.csv'
    }

    start_time = time.time()
    id_df = pd.read_csv(options['id_file'], usecols=['row'], dtype='float')
    meta_df = pd.read_csv(options['meta_file'])

    with open(options['corpus_file']) as corpus_file:
        corpus = pickle.load(corpus_file)
    lda = LdaMulticore.load(options['model_file'])

    if len(meta_df) != len(corpus):
        print ('Warning: Some documents may have been deleted during processing.\n')
        print ('metadata size - corpus size = ' + str(len(meta_df) - len(corpus)))

    topic_features = [to_dense(lda[bow], lda.num_topics) for bow in corpus]

    topic_colname = 'topic{0}'.format
    topic_colnames = [topic_colname(t+1) for t in xrange(lda.num_topics)]
    topic_df = pd.DataFrame.from_records(topic_features, columns=topic_colnames)
    with open('data\\topic_df.pkl', 'wb') as pkl_file:
        pickle.dump(topic_df, pkl_file)


    print ('topic size - id size = ' + str(len(id_df) - len(topic_df)))
    if len(id_df) != len(topic_df):
       raise Exception()

    topic_df = pd.concat([id_df, topic_df], axis=1)
    
    merged_df = pd.merge(meta_df, topic_df, on='row', how='right', sort=False)
    merged_df.to_csv(options['output_file'], index=False)

    end_time = time.time()
    print ('running time: ' + str((end_time - start_time)/60) + ' minutes')

if __name__ == '__main__':
    main()
