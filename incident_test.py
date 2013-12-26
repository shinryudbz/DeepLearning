import matplotlib.pyplot as plt
import math
import json
import numpy as np
import os
import time
import csv
from gensim.models.word2vec import *
import nltk
from scipy.stats import scoreatpercentile
import hungarian
import heapq

## Defaults ##
DEFAULT_VECTOR_SIZE = 3
DEFAULT_PERCENTILE_BUCKETS = 3
DEFAULT_BASE_PATH = ""
## Constants ##
FIELD_TYPE_NUMERIC = "numeric"
FIELD_TYPE_TEXT = "text"

"""" 
Example Schema:
{
	"dataset_path" : "iraq_incidents.csv",
	"fields" : {
		"Location":{
			"type" : "string"
		},
		"Target": {
			"type" : "string"
		},
		"Weapons": {
			"type" : "string"
		}, 
		"Reported_Minimum":{
			"type" : "numeric"
		}
	}
}
"""

### Functions for computing percentiles on a single value or over an entire schema ####

def convert_to_float(strValue):
	return float(strValue.replace(",", ""))

def compute_percentile(value, cutoffs):
	"""
	Given a value and a set of cutoffs returns the percentile within which the value lies
	:param value: the value for which to determine the percentile
	:param cutoffs: the cutoffs for 0th ...100th percentile (array of 10 ascending numbers)
	:return the percentile within which value lies
	"""
	if value < cutoffs[0]:
		return 0.0

	for i, cutoff in enumerate(cutoffs):
		if value < cutoff:
			return math.floor(100 * (float(i)/(len(cutoffs))))
			break
	return 100.0

def compute_schema_percentiles(schema):

	"""
	Given a schema with a dataset this function reads in the numeric fields and computes
	the percentile cutoffs for the numerric fields. It appends this information to the schema
	and returns it.
	:param schema: the schema for the dataset
	:return a schema with percentiles filled for each numeric field
	"""
	values = {}
	schemaFields = schema["fields"]
	for field in schemaFields:
		if schemaFields[field]["type"] == FIELD_TYPE_NUMERIC:
			values[field] = []
	
	def processRow(row, schemaFields, values):
		# cache the values for numerics
		for field in schemaFields:
			if schemaFields[field]["type"] == FIELD_TYPE_NUMERIC:
				values[field].append(convert_to_float(row[field]))

	read_dataset_as_key_values(schema, lambda x: processRow(x, schemaFields, values))
	if("buckets" in schema):
		NUM_BUCKETS = int(schema["buckets"])
	else:
		NUM_BUCKETS = DEFAULT_PERCENTILE_BUCKETS
	# compute percentiles
	for field in values:
		theList = np.array(values[field])
		theMedian = np.median(theList)
		oneSidedList = theList[:]
		oneSidedList[theList < theMedian] = 2*theMedian - theList[theList < theMedian]
		percentiles = []
		for i in xrange(0, NUM_BUCKETS):
			percentile =  math.floor(100 * (i / float(NUM_BUCKETS)))
			a = scoreatpercentile(oneSidedList, percentile)
			percentiles.append(a)
		schemaFields[field]["percentile"] = percentiles
	schema["fields"] = schemaFields
	return schema

### Functions for reading csv format data ####

def read_dataset_as_key_values(schema, callback,  updateStr=None, updateModulo = 1000):
	"""
	Reads in the csv at csv_path. The first row is assumed to be keys 
	callback is called with a dictionary with keys corresponding to column names 
	and strings contianing values
	:param csv_path: the path to the csv file
	:param callback: a function taking in a single parameter (dictionary of key values for the row)
	:param updateStr: (optional) a string that will display as the file reads each row
	:param updateModulo: (optional) the number of rows to skip before showing the message
	"""
	csv_path = os.path.join(DEFAULT_BASE_PATH, schema["dataset_path"])
	if "delimiter" in schema:
		delimiterStr = '\t'
	else:
		delimiterStr = ","
	fields = None
	num = 0
	with open(csv_path, 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=delimiterStr, quotechar='"')
		for row in reader:
			if not fields:
				fields = {}
				for i, field in enumerate(row):
					fields[field] = i
			else:
				keyValue = {}
				for field in fields:
					value = ''.join([x for x in row[fields[field]] if ord(x) < 128])
					keyValue[field] = value
				callback(keyValue)
				num += 1
				if (updateStr and num % updateModulo == 0):
					print updateStr + " " + str(num)


### Functions for reading and writing sentences from a schema ####

def read_sentences(schema, callback, search_sentences = False, update_message=None):
	"""
	Reads the sentences for the given schema and passes them back to the callback function
	in the object given to the callback each field (key) maps to a set of sentences
	:param schema: the dataset schemaFields
	:param callback: the function which receives the sentences read
	"""
	if(search_sentences):
		path = search_sentence_file_path(schema)
	else:
		path = sentence_file_path(schema)
	with open(path, "r") as f:
		i = 0
		while True:
			l = f.readline()
			i = i+1
			if not l:
				break
			callback(json.loads(l.rstrip()))
			if(update_message and i%1000 == 0):
				print update_message + " " +  str(i)


def build_sentences(schema):
	"""
	Generates a sentence file for the schema. 
	:param schema: the dataset schema
	"""
	with open(sentence_file_path(schema), "w") as output:
		def processKeyValue(keyValue, output, schema):
			sentencesByKey = {}
			for key in keyValue:
				words = generate_field_features(schema, key, keyValue[key])
				if(len(words)):
					sentencesByKey[key] = words
			output.write(json.dumps(sentencesByKey)+"\n")
			
		read_dataset_as_key_values(schema, lambda x : processKeyValue(x, output, schema), "reading")

def build_search_sentences(schema):
	# build list of validWords in model:
	validWords = map (lambda x : x.rstrip().split(" ")[0], open(weight_matrix_path(schema), "r").readlines())
	with open(search_sentence_file_path(schema), "w") as output:
		def processKeyValue(keyValue, output, schema, validWords):
			sentencesByKey = {}
			for key in keyValue:
				words = generate_field_features(schema, key, keyValue[key])
				if(len(words)):
					sentencesByKey[key] = filter( lambda x : x in validWords, words)
			sentencesByKey["original_data"] = keyValue
			output.write(json.dumps(sentencesByKey)+"\n")
			
		read_dataset_as_key_values(schema, lambda x : processKeyValue(x, output, schema, validWords), "reading")
### functions for getting paths for various generated files from within the schema ###

def model_path(schema):
	return os.path.join(DEFAULT_BASE_PATH, os.path.dirname(schema["dataset_path"]), os.path.basename(schema["dataset_path"].split(".")[0] + "_model.out"))

def weight_matrix_path(schema):
	return os.path.join(DEFAULT_BASE_PATH, os.path.dirname(schema["dataset_path"]), os.path.basename(schema["dataset_path"].split(".")[0] + "_weight.out"))

def sentence_file_path(schema):
	return os.path.join(DEFAULT_BASE_PATH, os.path.dirname(schema["dataset_path"]), os.path.basename(schema["dataset_path"].split(".")[0] + "_sentences.out"))

def search_sentence_file_path(schema):
	return os.path.join(DEFAULT_BASE_PATH, os.path.dirname(schema["dataset_path"]), os.path.basename(schema["dataset_path"].split(".")[0] + "_search_sentences.out"))



### functions for manipulating a piece of raw data into sentences ###
def convert_key_value_to_sentence(schema, keyValues, fieldsToRead):
	"""
	Given a set of raw key values read from base data returns a sentence which can be fed to a Word2Vec model
	:param schema: dataset schema
	:param keyValues: the key values (read from a csv) to convert to sentences
	:param fieldsToRead: the keys to actually read (others will be ignored)
	:return a sentence usable by a Word2Vec model
	"""
	features = []
	for field in fieldsToRead:
		value = keyValues[field]
		features += generate_field_features(schema,field, value)
	return features


def merge_sentences_to_single_sentence(keyValue, fieldsToRead):
	"""
	Given a keyValue where values are already sentences (keyed by field) it converts
	the input into a single sentence by pulling fields from fieldsToRead
	:param keyValue: the map from fields to sentences
	:param fieldsToRead: the set of keys to actually useable
	:return a single sentences with fields from fieldsToRead
	"""
	ret = []
	for field in fieldsToRead:
		if(field in keyValue):
			ret +=  keyValue[field]
	return ret

def generate_field_features(schema, field, value):
	"""
	Given a field value converts it to a sentence that is useable by the model
	:param schema: dataset schema
	:param field: fieldname in the schema for value
	:param value: actual raw value
	:return a sentence usable by a Word2Vec model
	"""
	if not field in schema["fields"]:
		return []

	if(schema["fields"][field]["type"] == FIELD_TYPE_NUMERIC):
		return [field + "_" + str(compute_percentile(convert_to_float(value), schema["fields"][field]["percentile"]))+"_percentile"]
	else:
		rawWords = map(lambda x : x.lower() , nltk.word_tokenize(value))
		featuresNonUnicode = []
		for word in rawWords:
			try:
				raw = word.encode('ascii', 'ignore')
				featuresNonUnicode.append(raw)
			except:
				continue
		return featuresNonUnicode


### methods for training a model ###
def train_model(schema,fieldsToRead = None):
	"""
	Given a schema and vectorSize trains the model.
	:param schema: dataset schema
	:param vectorSize: size of feature vectors to generate
	:param fieldsToRead: the keys to actually train on (others will be ignored)
	:return a Word2Vec model
	"""
	if not fieldsToRead:
		fieldsToRead = schema["fields"].keys()

	if("vector_size" in schema):
		vectorSize = schema["vector_size"]
	else:
		vectorSize = DEFAULT_VECTOR_SIZE

	sentences = []
	# build sentences:
	print "Building Feature vectors..."

	read_sentences(schema, lambda x : sentences.append(merge_sentences_to_single_sentence(x, fieldsToRead)))
	print "Read " + str(len(sentences)) + " documents"
	print "Training Model..."
	modelPath = model_path(schema)
	weightMatrixPath = weight_matrix_path(schema)
	model = Word2Vec(sentences, size=vectorSize, window=5, min_count=1, workers=4)
	model.save(modelPath)
	model.save_word2vec_format(weightMatrixPath)
	print "Finished training"
	return model

## method for comparing and searching documents
def compare_docs(sentenceA, sentenceB, model, similarityCache, worstInQueue):
	"""
	Document comparison algorithm
	Given  documents A,B and |A|<|B|
	assign an edge between all points in A to all points in B with weight = 1 / similarity(p1,p2)
	run hungarian algorithm to find maximal matching (this yields a subset of A's points)
	:param sentenceA: a sentence of words that are contained in the model
	:param sentenceB: another sentence of words that are contained in the model
	:param model: a Word2Vec model
	:return sum of weights as score
	"""
	MAX_VAL = 999999
	lA = len(sentenceA)
	n = max(lA, len(sentenceB))

	if(lA < n):
		shorter = sentenceA
		longer = sentenceB
	else:
		shorter = sentenceB
		longer = sentenceA
	
	lS = len(shorter)

	# compute distance matrix from A => B
	mat =  np.zeros((n, n))
	mat.fill(MAX_VAL)
	mostSimilar = None
	
	for i in xrange(0, lS):
		for j in xrange(0,n):
			key = (shorter[i], longer[j])
			similarityVal = 0
			try:
				similarityVal = similarityCache[key]
			except:
				# TODO: GET RID OF TRY - CATCH!
				similarityVal = model.similarity(shorter[i], longer[j])
				similarityCache[key] = similarityVal
			if (not mostSimilar or similarityVal > mostSimilar):
				mostSimilar = similarityVal
			mat[i][j] = -similarityVal; # cost is inverse of similarity

	# early abort hungarian if there is no way we are in the top k:
	if(mostSimilar and mostSimilar * len(shorter) < worstInQueue):
		return -1
	# run hungarian algorithm on cost matrix
	matches = hungarian.lap(mat)[0]

	# sum over minimal distance matching
	total = 0
	for i in xrange(0, len(shorter)):
		key = (shorter[i], longer[matches[i]])
		total += similarityCache[key];
	return total


def run_point_cloud_search(positiveDoc, negativeDoc, schema, model, validWords, fieldsToCompare = None, numResults = 10):
	"""
	Point cloud search algoirthm. Given two key_value documents (positiveDoc, negativeDoc) it finds documents
	similar to positive while also being dissimilar to negative.
	:param positiveDoc: a json document which is "positive"
	:param negativeDoc: a json document which is "negative"
	:param schema: a schema for the dataset
	:param model: a Word2Vec model
	:param fieldsToCompare: (optional) fields to compare within the json doc
	:param numResults: (optional) number of results to return
	:return list of best matches
	"""

	if not fieldsToCompare:
		fieldsToCompare = schema["fields"].keys()
	# convert to sentence if doc is not a sentence
	
	if type(positiveDoc) == type({}):
		docPositive = convert_key_value_to_sentence(schema, doc, fieldsToCompare)
	else:
		docPositive = positiveDoc
		docPositive = filter(lambda x : x in validWords, docPositive)

	if type(negativeDoc) == type({}):
		docNegative = convert_key_value_to_sentence(schema, doc, fieldsToCompare)
	else:
		docNegative = negativeDoc
		docNegative = filter(lambda x : x in validWords, docNegative)

	
	print "Similar words:"
	print model.most_similar(docPositive, docNegative)

	# build a search queue using heapq
	heap = []
	similarityCache = {}
	worst = { "value" : None }
	# scan through dataset
	def compareValue(docPositive, docNegative, docToCompare, heap, model, fieldsToCompare, similarityCache, worst):
		docToCompareFiltered = merge_sentences_to_single_sentence(docToCompare, fieldsToCompare)
		score = compare_docs(docPositive, docToCompareFiltered, model, similarityCache, worst["value"])
		if(docNegative):
			score -= compare_docs(docNegative, docToCompareFiltered, model, similarityCache, worst["value"])

		if(worst["value"] == None or (worst["value"] < score)):
			heapq.heappush(heap, (score,docToCompare))
		
		# pop the lowest score if we've gotten too many items
		if len(heap) > numResults:
			worst["value"] = heapq.heappop(heap)[0]

	read_sentences(schema, lambda x : compareValue(docPositive, docNegative, x, heap, model, fieldsToCompare, similarityCache, worst), True)
	ret = []
	for i in xrange(0, numResults):
		ret.append(heapq.heappop(heap))
	ret.reverse()
	return ret

def plot_schema_vectors(schema, model):
	xs = []
	ys = []
	fig, ax = plt.subplots()
	vectorList = map(lambda x : x.rstrip().split(' ')[0], open(weight_matrix_path(schema), "r").readlines())

	words = vectorList[:100]
	annotations = []
	if(len(words[0]) != 2):
		print "Vectors must be of dimension 2 to plot!"
		return
	for word in words:
		try:
			w = model[word]
			xs.append(w[0])
			ys.append(w[1])
			annotations.append(word)
		except:
			continue
	
	ax.scatter(xs, ys)
	for i, txt in enumerate(annotations):
		ax.annotate(txt, (xs[i],ys[i]))
	plt.show()


if __name__ == '__main__':
	
	if(len(sys.argv)>1):
		path = sys.argv[1]
	else:
		path = 'data/iris/iris_schema.json'

	DEFAULT_BASE_PATH = os.path.dirname(path)

	json_data = open(path).read()
	schema = json.loads(json_data)
	schema = compute_schema_percentiles(schema)

	# check if sentence dump has been created:
	try:
		open(sentence_file_path(schema), "r")
	except:
		# build sentence dump if load fails:
		build_sentences(schema)

	# check if model has been trained:
	try:
		# try loading model
		model =  Word2Vec.load_word2vec_format(weight_matrix_path(schema), binary=False)
	except:
		# otherwise compute it:
		model = train_model(schema)

	plot_schema_vectors(schema,model)
	# check if search sentences have been made :
	try:
		# try loading model
		open(search_sentence_file_path(schema), "r")
	except:
		# build sentence dump if load fails:
		build_search_sentences(schema)

	# run the point cloud search:
	validWords = map (lambda x : x.rstrip().split(" ")[0], open(weight_matrix_path(schema), "r").readlines())
	while True:
		start1 = time.time();
		pos = raw_input("Positive features?").split()
		neg = raw_input("Negative features?").split()

		ret = run_point_cloud_search(pos, neg, schema, model, validWords, None, 10)
		start2 = time.time();

		for val in ret:

			keyvals = val[1]["original_data"]
			for key in keyvals:
				if key in schema["fields"] and schema["fields"][key]["type"] == "numeric":
					print key + " : " + str(keyvals[key]) + " (" + str(compute_percentile(convert_to_float(keyvals[key]),schema["fields"][key]["percentile"])) + " percentile)"
				else:
					print key + " : " + str(keyvals[key])
			print "Score : " + str(val[0])
			print ""
			print "------"
