from gensim.models.word2vec import *
import hungarian


def convert_to_word_array(doc, fieldsToCompare):
	#TODO
	return ["review_count_90.0_percentile", "votes_cool_80.0_percentile", "love"];

def compare_docs(docA, docB, fieldsToCompare, model):
	wordsA = ["review_count_90.0_percentile", "food", "love"] #convert_to_word_array(docA, fieldsToCompare)
	wordsB = ["review_count_90.0_percentile", "phoenix", "food"] #convert_to_word_array(docA, fieldsToCompare)

	
	n = max(len(wordsA), len(wordsB))

	if(len(wordsA) < n):
		shorter = wordsA
		longer = wordsB
	else:
		shorter = wordsB
		longer = wordsA

	# compute distance matrix from A=>B
	mat = []
	for i in xrange(0, n):
		row = [0 for m in xrange(0,n)];
		for j in xrange(0,n):
			if(i >= len(shorter)):
				row[j] = float("inf");
			else:
				row[j] = abs(1.0 / model.similarity(shorter[i], longer[j]))
		mat.append(row)

	# run hungarian algorithm on cost matrix
	scores = hungarian.lap(mat)[1]

	# sum over minimal distance matching
	total = 0
	for i in xrange(0, len(shorter)):
		total += model.similiarity(1.0 / shorter[scores[i]], longer[i])
	return total



if __name__ == '__main__':
	print "Loading Model"
	model =  Word2Vec.load_word2vec_format('model/wgt.out', binary=False)
	print "Done"
	#print model.most_similar(positive=['review_count_90.0_percentile'], topn=100)

	#Basic algorithm (to start):
	# Given  doc A with |A| = i and doc B with |B| = j and i<j
	# pick a set of i unique points in B which result in the lowest total distance
	# (assign an edge between all points in A to all points in B with weight= 1 / distance)
	# run hungarian algorithm to find maximal matching (this yields subset of A's points)
	# return sum of weights as score
	# 
	# So function looks like
	# Compare(docA, docB, fieldsToCompare, model)
	# Need function which given JSON of docA, docB, and fieldsToCompare outputs
	# list of words that we can stick into the model
	
	compare_docs(None, None, None, model)

