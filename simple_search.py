from gensim.models.word2vec import *
import hungarian
import heapq




def compute_percentile(value, cutoffs):
	if value < cutoffs[0]:
		return 0.0

	for i, cutoff in enumerate(cutoffs):
		if value < cutoff:
			return 100 * (float(i)/(len(cutoffs)))
			break
	return 100.0

def convert_json_to_sentence(doc, schema, fieldsToCompare):
	sentence = []
	for field in fieldsToCompare:
		if(not field in schema):
			print "Invalid field specified : " + field + " .... skipping"
			continue

		else:
			fieldMetaData = schema[field]
			if(fieldMetaData["type"] is "string"):
				sentence += doc[field]
			else:
				# compute the percentile and create a word:
				sentence += [field + "_"+ str(compute_percentile(doc[field], fieldMetaData["percentile"])) + "_percentile"];
	return sentence

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

def run_search(doc, dataset, schema, fieldsToCompare, numResults):
	docA = convert_json_to_sentence(doc, schema, fieldsToCompare);
	
	# build a search queue using heapq
	heap = []
	
	# scan through dataset
	for docToCompare in dataset:
		docB = convert_json_to_sentence(docToCompare, schema, fieldsToCompare)
		score = compare_docs(docA, docB,)
		heapq.heappush(heap, (score,docB))

		# pop the lowest score if we've gotten too many items
		if len(heap) > numResults:
			heapq.heappop(heap)
	
	ret = []
	for i in xrange(0, numResults):
		ret.append(heapq.heappop(heap)[1])
	return ret

#  schema = {"name" : "review_count", "type" : "numeric", "percentiles":[1,23,3,44,88,99,100...10]}

if __name__ == '__main__':
	print "Loading Model"

	# percentile array contains:
	#  [10th percentile start, 20th percentile start ... 90th percentile start, 100th percentile start]
	# example:
	print compute_percentile( 99, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
	

	"""
	model =  Word2Vec.load_word2vec_format('model/wgt.out', binary=False)
	print "Done"
	#print model.most_similar(positive=['review_count_90.0_percentile'], topn=100)

	# TODO: Load dataset in to json
	

	sentenceA = ["review_count_90.0_percentile", "food", "love"];
	sentenceB = ["review_count_90.0_percentile", "phoenix", "food", "blah", "love"]
	compare_docs(wordsA, wordsB, model)


"""