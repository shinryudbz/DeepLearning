from gensim.models.word2vec import *



if __name__ == '__main__':
	print "Loading Model"
	model =  Word2Vec.load_word2vec_format('model/wgt.out', binary=False)
	print "Done"
	print model.most_similar(positive=['votes_cool_70.0_percentile'], negative=['votes_funny_60.0_percentile'], topn=100)
	print model.most_similar(positive=['votes_cool_70.0_percentile', 'votes_funny_60.0_percentile'], topn=100)
