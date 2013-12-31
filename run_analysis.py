import matplotlib.pyplot as plt
import math
import json
import numpy as np
import os
import time
import scipy.spatial.distance
import csv
from gensim.models.word2vec import *
import nltk
from scipy.stats import scoreatpercentile
import heapq

## Defaults ##
DEFAULT_VECTOR_SIZE = 2
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
	if(len(values) == 0):
		return schema
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
					field = field.strip()
					fields[field] = i
			else:
				try:
					keyValue = {}
					for field in fields:
						value = ''.join([x for x in row[fields[field]] if ord(x) < 128])
						keyValue[field] = value
					callback(keyValue)
					num += 1
					if (updateStr and num % updateModulo == 0):
						print updateStr + " " + str(num)
				except:
					print "Invalid row"
					print row
					continue


### Functions for reading and writing sentences from a schema ####

def read_sentences(schema, callback, update_message=None):
	"""
	Reads the sentences for the given schema and passes them back to the callback function
	in the object given to the callback each field (key) maps to a set of sentences
	:param schema: the dataset schemaFields
	:param callback: the function which receives the sentences read
	"""
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

### functions for getting paths for various generated files from within the schema ###

def model_path(schema):
	return os.path.join(DEFAULT_BASE_PATH, os.path.dirname(schema["dataset_path"]), os.path.basename(schema["dataset_path"].split(".")[0] + "_model.out"))

def weight_matrix_path(schema):
	return os.path.join(DEFAULT_BASE_PATH, os.path.dirname(schema["dataset_path"]), os.path.basename(schema["dataset_path"].split(".")[0] + "_weight.out"))

def sentence_file_path(schema):
	return os.path.join(DEFAULT_BASE_PATH, os.path.dirname(schema["dataset_path"]), os.path.basename(schema["dataset_path"].split(".")[0] + "_sentences.out"))


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

def transpose_sentences(sentences):
	seen  = {}
	for i, sentence in enumerate(sentences):
		for word in sentence:
			if not word in seen:
				seen[word] = []
			seen[word].append("sentence_"+str(i))
	return seen.values()

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
	sentences = transpose_sentences(sentences)
	model = Word2Vec(sentences, size=vectorSize, window=5, min_count=1, workers=4)
	model.save(modelPath)
	model.save_word2vec_format(weightMatrixPath)
	print "Finished training"
	return model

def compute_similarity(model, word, positives, negatives=[]):
	vec1 = np.array(model[word])
	score = 0
	for p in positives:
		vec2 = np.array(model[p])
		score +=  1.0 / len(positives) *  np.linalg.norm(vec2-vec1)

	for n in negatives:
		vec2 = np.array(model[n])
		score -= 1.0 / len(negatives) * np.linalg.norm(vec2-vec1)
	
	return score

def most_similar(wordList, model, positives, negatives = None, numToReturn=100):
	heap = []
	for word in wordList:
		score = compute_similarity(model, word, positives, negatives)
		heapq.heappush(heap, (score,word))
	ret = []

	while(len(heap) > 0 and len(ret) < numToReturn):
		ret.append(heapq.heappop(heap))
	return ret

def plot_schema_vectors(schema, model):
	"""
	Plots the first 100 word vectors for the given model
	"""
	vectorList = map(lambda x : x.rstrip().split(' ')[0], open(weight_matrix_path(schema), "r").readlines())[1:]
	words = vectorList[:500]
	
	if(len(model[words[0]]) != 2):
		print "Vectors must be of dimension 2 to plot! ... Aborting (vector dim is : " + str(len(model[words[0]])) + ")"
		return
	xs = []
	ys = []
	fig, ax = plt.subplots()
	

	
	annotations = []
	
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

	print "Loading : " + path

	print "Computing Schema Numeric Percentiles..."
	schema = compute_schema_percentiles(schema)
	print "Done"

	# check if model has been trained:
	try:
		print "Loading model"
		# try loading model
		model =  Word2Vec.load_word2vec_format(weight_matrix_path(schema), binary=False)
		print "Done"
	except:
		print "Need to build model..."
		# check if sentence dump has been created:
		try:
			print "Loading training sentences"
			open(sentence_file_path(schema), "r")
		except:
			# build sentence dump if load fails:
			print "Need to build training sentences"
			print "Building training sentences"
			build_sentences(schema)
			print "Done"
		# otherwise compute it:
		print "Training Model"
		model = train_model(schema)
		print "Done"

	plot_schema_vectors(schema,model)
	
	# run the point cloud search:
	print "Loading valid words"
	validWords = map (lambda x : x.rstrip().split(" ")[0], open(weight_matrix_path(schema), "r").readlines())[1:]
	while True:
		pos = raw_input("Positive features?").split()
		neg = raw_input("Negative features?").split()
		ret = most_similar(validWords, model, pos, neg);
		for r in ret:
			print r
		
		print model.most_similar(pos, neg)
"""
		start1 = time.time();
		pos = raw_input("Positive features?").split()
		neg = raw_input("Negative features?").split()

		ret = run_point_cloud_search(pos, neg, schema, model, validWords, None, 10)
		start2 = time.time();
		print "Time: " + str(start2-start1)
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
"""