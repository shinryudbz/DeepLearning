from pandas import *
import json
from pprint import pprint
import numpy as np
import os
import csv
from gensim.models.word2vec import *
from gensim.utils import simple_preprocess
from nltk.stem import PorterStemmer
from sets import Set
import time
import copy
import math
import json
import nltk
from scipy.stats import scoreatpercentile

FIELD_TYPE_NUMERIC = "numeric"
FIELD_TYPE_TEXT = "text"

def compute_percentile(value, cutoffs):
	if value < cutoffs[0]:
		return 0.0

	for i, cutoff in enumerate(cutoffs):
		if value < cutoff:
			return 100 * (float(i)/(len(cutoffs)))
			break
	return 100.0

def compute_schema_percentiles(schema):
	fields = None
	values = {}
	csv_path = schema["dataset_path"]
	schemaFields = schema["fields"]
	for field in schemaFields:
		if schemaFields[field]["type"] == FIELD_TYPE_NUMERIC:
			values[field] = []
	with open(csv_path, 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in reader:
			if not fields:
				fields = {}
				for i, field in enumerate(row):
					fields[field] = i

			else:
				# cache the values for numerics
				for field in schemaFields:
					if field in fields and schemaFields[field]["type"] == FIELD_TYPE_NUMERIC:
						values[field].append(float(row[fields[field]]));
	# compute percentiles
	for field in values:
		theList = np.array(values[field])
		theMedian = np.median(theList)
		oneSidedList = theList[:]
		oneSidedList[theList < theMedian] = 2*theMedian - theList[theList < theMedian]
		percentiles = []
		for i in xrange(0, 10):
			percentile =  10 * i
			a = scoreatpercentile(oneSidedList, percentile) - theMedian;
			percentiles.append(a)
		schemaFields[field]["percentile"] = percentiles
	schema["fields"] = schemaFields
	return schema

def model_path(schema):
	return os.path.join(os.path.dirname(schema["dataset_path"]), os.path.basename(schema["dataset_path"].split(".")[0] + "_model.out"))

def weight_matrix_path(schema):
	return os.path.join(os.path.dirname(schema["dataset_path"]), os.path.basename(schema["dataset_path"].split(".")[0] + "_weight.out"))


STEMMER = PorterStemmer()

def generate_field_features(schema, field, value):
	global STEMMER
	if(schema["fields"][field]["type"] == FIELD_TYPE_NUMERIC):
		return [field + "_" + str(compute_percentile(float(value), schema["fields"][field]["percentile"]))+"_percentile", field+"_"+str(value)+"_value"]
	else:
		return nltk.word_tokenize(value)


def train_model(schema, vectorSize, fieldsToRead = None):
	if not fieldsToRead:
		fieldsToRead = schema["fields"].keys()

	sentences = []
	# build sentences:
	print "Building Feature vectors..."
	fields = None
	csv_path = schema["dataset_path"]
	num = 0
	with open(csv_path, 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in reader:
			features = []
			num += 1
			if num % 1000 == 0:
				print num
			if not fields:
				fields = {}
				for i, field in enumerate(row):
					fields[field] = i
			else:
				for field in fieldsToRead:
					value = row[fields[field]]
					features += generate_field_features(schema,field, value)
			featuresNonUnicode = []
			for feature in features:
				try:
					raw = feature.encode('ascii', 'ignore')
					featuresNonUnicode.append(raw)
				except:
					continue
			sentences.append(featuresNonUnicode)

	print "Generated " + str(len(sentences)) + " documents"
	print "Training Model..."
	modelPath = model_path(schema)
	weightMatrixPath = weight_matrix_path(schema)
	model = Word2Vec(sentences, size=vectorSize, window=5, min_count=5, workers=4)
	model.save(modelPath)
	model.save_word2vec_format(weightMatrixPath)



json_data = open(sys.argv[1]).read()
schema = json.loads(json_data)
schema = compute_schema_percentiles(schema)
train_model(schema, 100)

"""
stemmer = PorterStemmer()

stemmer.stem(word)

print "Training model..."
model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
model.save(modelPath)
model.save_word2vec_format(wgtMatrixPath)
"""