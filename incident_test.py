import pylab
import json
import numpy as np
import os
import time
import csv
from gensim.models.word2vec import *
import nltk
from scipy.stats import scoreatpercentile
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import hungarian
import heapq

FIELD_TYPE_NUMERIC = "numeric"
FIELD_TYPE_TEXT = "text"
STEMMER = PorterStemmer()


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
			return 100 * (float(i)/(len(cutoffs)))
			break
	return 100.0

def read_csv_as_key_values(csv_path, callback, updateStr = None, updateModulo = 1000):
	"""
	Reads in the csv at csv_path. The first row is assumed to be keys 
	callback is called with a dictionary with keys corresponding to column names 
	and strings contianing values
	:param csv_path: the path to the csv file
	:param callback: a function taking in a single parameter (dictionary of key values for the row)
	:param updateStr: (optional) a string that will display as the file reads each row
	:param updateModulo: (optional) the number of rows to skip before showing the message
	"""
	fields = None
	num = 0
	with open(csv_path, 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in reader:
			if not fields:
				fields = {}
				for i, field in enumerate(row):
					fields[field] = i
			else:
				keyValue = {}
				for field in fields:
					value = row[fields[field]]
					keyValue[field] = value
				callback(keyValue)
				num += 1
				if (updateStr and num % updateModulo == 0):
					print updateStr + " " + str(num)

def compute_schema_percentiles(schema):
	"""
	Given a schema with a dataset this function reads in the numeric fields and computes
	the percentile cutoffs for the numerric fields. It appends this information to the schema
	and returns it.
	:param schema: the schema for the dataset
	:return a schema with percentiles filled for each numeric field
	"""
	values = {}
	csv_path = schema["dataset_path"]
	schemaFields = schema["fields"]
	for field in schemaFields:
		if schemaFields[field]["type"] == FIELD_TYPE_NUMERIC:
			values[field] = []
	
	def processRow(row, schemaFields, values):
		# cache the values for numerics
		for field in schemaFields:
			if schemaFields[field]["type"] == FIELD_TYPE_NUMERIC:
				values[field].append(float(row[field]))

	read_csv_as_key_values(csv_path, lambda x: processRow(x, schemaFields, values))

	# compute percentiles
	for field in values:
		theList = np.array(values[field])
		theMedian = np.median(theList)
		oneSidedList = theList[:]
		oneSidedList[theList < theMedian] = 2*theMedian - theList[theList < theMedian]
		percentiles = []
		for i in xrange(0, 10):
			percentile =  10 * i
			a = scoreatpercentile(oneSidedList, percentile) - theMedian
			percentiles.append(a)
		schemaFields[field]["percentile"] = percentiles
	schema["fields"] = schemaFields
	return schema

def model_path(schema):
	return os.path.join(os.path.dirname(schema["dataset_path"]), os.path.basename(schema["dataset_path"].split(".")[0] + "_model.out"))

def weight_matrix_path(schema):
	return os.path.join(os.path.dirname(schema["dataset_path"]), os.path.basename(schema["dataset_path"].split(".")[0] + "_weight.out"))

def sentence_file_path(schema):
	return os.path.join(os.path.dirname(schema["dataset_path"]), os.path.basename(schema["dataset_path"].split(".")[0] + "_sentences.out"))


HUNG_TIME = 0
MAT_TIME = 0
SIMILARITY_TIME = 0
def compare_docs(sentenceA, sentenceB, model, similarityCache, worstInQueue):
	global HUNG_TIME
	global MAT_TIME
	global SIMILARITY_TIME
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
	n = max(len(sentenceA), len(sentenceB))

	if(len(sentenceA) < n):
		shorter = sentenceA
		longer = sentenceB
	else:
		shorter = sentenceB
		longer = sentenceA

	# compute distance matrix from A => B
	mat =  np.zeros((n, n))
	mostSimilar = None
	for i in xrange(0, n):
		for j in xrange(0,n):
			if(i >= len(shorter)):
				mat[i][j] = 99999
			else:
				mstart = time.time()
				key = (shorter[i], longer[j])
				similarityVal = None
				if(not key in similarityCache):
					try:
						similarityVal = model.similarity(shorter[i], longer[j])
					except:
						similarityVal = 99999
					similarityCache[key] = similarityVal
				else:
					similarityVal = similarityCache[key]
				if (not mostSimilar or similarityVal > mostSimilar):
					mostSimilar = similarityVal;
				mend = time.time()
				SIMILARITY_TIME += mend-mstart
				mat[i][j] = abs(1.0 / similarityVal)
	# early abort hungarian if there is no way we are in the top k:
	if(mostSimilar and mostSimilar * len(shorter) < worstInQueue):
		return -1

	# run hungarian algorithm on cost matrix
	
	MAT_TIME +=(end-start)
	start = time.time()
	matches = hungarian.lap(mat)[0]
	end = time.time()
	HUNG_TIME +=(end-start)
	# sum over minimal distance matching
	total = 0
	for i in xrange(0, len(shorter)):
		key = (shorter[i], longer[matches[i]])
		total += similarityCache[key];
	return total

def run_point_cloud_search(positiveDoc, negativeDoc, schema, model, fieldsToCompare = None, numResults = 100):
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

	if type(negativeDoc) == type({}):
		docNegative = convert_key_value_to_sentence(schema, doc, fieldsToCompare)
	else:
		docNegative = negativeDoc
	
	# build a search queue using heapq
	heap = []
	similarityCache = {}
	worst = { "value" : None }
	# scan through dataset
	def compareValue(docPositive, docNegative, docToCompare, heap, model, fieldsToCompare, similarityCache, worst):
		docToCompareFiltered = keys_to_single_sentence(docToCompare, fieldsToCompare)
		score = compare_docs(docPositive, docToCompareFiltered, model, similarityCache, worst["value"])
		if(docNegative):
			score -= compare_docs(docNegative, docToCompareFiltered, model, similarityCache, worst["value"])
		heapq.heappush(heap, (score,docToCompare))
		# pop the lowest score if we've gotten too many items
		if len(heap) > numResults:
			worst["value"] = heapq.heappop(heap)[0]

	read_sentences(schema, lambda x : compareValue(docPositive, docNegative, x, heap, model, fieldsToCompare, similarityCache, worst), "Searching")
	ret = []
	for i in xrange(0, numResults):
		ret.append(heapq.heappop(heap))
	ret.reverse()
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

	global STEMMER
	if(schema["fields"][field]["type"] == FIELD_TYPE_NUMERIC):
		return [field + "_" + str(compute_percentile(float(value), schema["fields"][field]["percentile"]))+"_percentile", field+"_"+str(value)+"_value"]
	else:
		rawWords = map(lambda x : STEMMER.stem(x.lower()) , nltk.word_tokenize(value))
		featuresNonUnicode = []
		for word in rawWords:
			try:
				raw = word.encode('ascii', 'ignore')
				featuresNonUnicode.append(raw)
			except:
				continue
		return featuresNonUnicode

def convert_key_value_to_sentence(schema, keyValues, fieldsToRead):
	"""
	Given a set of key values returns a sentence which can be fed to a Word2Vec model
	:param schema: dataset schema
	:param keyValues: the key values to convert to sentences
	:param fieldsToRead: the keys to actually read (others will be ignored)
	:return a sentence usable by a Word2Vec model
	"""
	features = []
	for field in fieldsToRead:
		value = keyValues[field]
		features += generate_field_features(schema,field, value)
	return features

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
			
		read_csv_as_key_values(schema["dataset_path"], lambda x : processKeyValue(x, output, schema), "reading")

def read_sentences(schema, callback, update_message=None):
	"""
	Reads the sentences for the given schema and passes them back to the callback function
	in the object given to the callback each field (key) maps to a set of sentences
	:param schema: the dataset schemaFields
	:param callback: the function which receives the sentences read
	"""
	with open(sentence_file_path(schema), "r") as f:
		i = 0
		while True:
			l = f.readline()
			i = i+1
			if not l:
				break
			callback(json.loads(l.rstrip()))
			if(update_message and i%1000 == 0):
				print update_message + " " +  str(i)

def keys_to_single_sentence(keyValue, fieldsToRead):
	"""
	Given a keyValue where values are sentences it converts
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

def train_model(schema, vectorSize, fieldsToRead = None):
	"""
	Given a schema and vectorSize trains the model.
	:param schema: dataset schema
	:param vectorSize: size of feature vectors to generate
	:param fieldsToRead: the keys to actually train on (others will be ignored)
	:return a Word2Vec model
	"""
	if not fieldsToRead:
		fieldsToRead = schema["fields"].keys()

	sentences = []
	# build sentences:
	print "Building Feature vectors..."

	read_sentences(schema, lambda x : sentences.append(keys_to_single_sentence(x, fieldsToRead)))
	print "Read " + str(len(sentences)) + " documents"
	print "Training Model..."
	modelPath = model_path(schema)
	weightMatrixPath = weight_matrix_path(schema)
	model = Word2Vec(sentences, size=vectorSize, window=5, min_count=5, workers=4)
	model.save(modelPath)
	model.save_word2vec_format(weightMatrixPath)
	print "Finished training"
	return model



json_data = open(sys.argv[1]).read()
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
	model = train_model(schema, 50)


# run the point cloud search:
ret = run_point_cloud_search(["Reported_Minimum_100.0_percentile","children","child"], ["bomb", "car"], schema, model)
print "Hungarian algorithm time"
print HUNG_TIME
print "Similarity matrix computation time"
print MAT_TIME
print "Similarity computation time"
print SIMILARITY_TIME
"""
xs = []
ys = []
zs = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vectorList = map(lambda x : x.rstrip().split(' ')[0], open(weight_matrix_path(schema), "r").readlines())

print vectorList
words = map( lambda x : STEMMER.stem(x.lower()), vectorList[:100])
for word in words:
	try:
		w = model[word]
		xs.append(w[0])
		ys.append(w[1])
		zs.append(w[2])
	except:
		continue
sc = ax.scatter(xs, ys, zs)

labels = []
for word in words:
	try:
		x2, y2, _ = proj3d.proj_transform(model[word][0],model[word][1],model[word][2], ax.get_proj())
		label = ax.annotate(word, xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
		labels.append((label, model[word]))
	except:
		continue

def update_position(e):
	for labelArr in labels:
		x2, y2, _ = proj3d.proj_transform(labelArr[1][0], labelArr[1][1], labelArr[1][2], ax.get_proj())
		labelArr[0].xy = x2,y2
		labelArr[0].update_positions(fig.canvas.renderer)
		fig.canvas.draw()

fig.canvas.mpl_connect('button_release_event', update_position)
pylab.show()
plt.show()
"""