from gensim.models.word2vec import *



if __name__ == '__main__':
	print "Loading Model"
	model =  Word2Vec.load_word2vec_format('model/wgt.out', binary=False)
	print "Done"
	print model.most_similar(positive=['review_count_90.0_percentile'], topn=100)