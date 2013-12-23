from gensim.models.word2vec import *
import hungarian





def convert_json_to_sentence(doc, fieldsToCompare):
	#TODO
	return ["review_count_90.0_percentile", "votes_cool_80.0_percentile", "love"];

# Document comparison algorithm
# Given  documents A,B and |A|<|B|
# assign an edge between all points in A to all points in B with weight = 1 / similarity(p1,p2)
# run hungarian algorithm to find maximal matching (this yields a subset of A's points)
# return sum of weights as score

def compare_docs(sentenceA, sentenceB, model):

	n = max(len(sentenceA), len(sentenceB))

	if(len(sentenceA) < n):
		shorter = sentenceA
		longer = sentenceB
	else:
		shorter = sentenceB
		longer = sentenceA

	# compute distance matrix from A => B
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
	matches = hungarian.lap(mat)[0]
	# sum over minimal distance matching
	total = 0
	for i in xrange(0, len(shorter)):
		total += model.similarity(shorter[i], longer[matches[i]])
	return total


def run_search(doc, dataset, numResults, fieldsToCompare):
	docA = convert_json_to_sentence(doc,fieldsToCompare);
	# build a search queue
	# scan through dataset
	for docToCompare in dataset:
		docB = convert_json_to_sentence(docToCompare, fieldsToCompare)
		score = compare_docs(docA, docB,)
		# push docB with score onto priority queue
		
		


if __name__ == '__main__':
	print "Loading Model"
	model =  Word2Vec.load_word2vec_format('model/wgt.out', binary=False)
	print "Done"
	#print model.most_similar(positive=['review_count_90.0_percentile'], topn=100)

	sentenceA = ["review_count_90.0_percentile", "food", "love"];
	sentenceB = ["review_count_90.0_percentile", "phoenix", "food", "blah", "love"]
	compare_docs(wordsA, wordsB, model)

