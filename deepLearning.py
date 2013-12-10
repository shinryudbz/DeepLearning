#-------------------------------------------------------------------------------
# Name:        deepLearning.py
# Purpose:     Experimenting with gensim
#
# Author:      Upal Hasan
#
# Created:     17/11/2013
# Copyright:   (c) Upal Hasan 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from pandas import *
import json
from pprint import pprint
import numpy
import os
from gensim.models.word2vec import *
from gensim.utils import simple_preprocess
from nltk.stem import PorterStemmer
from sets import Set
import time

# the base class from which all the other classes inherit from. it contains
# a lot of useful methods that the other classes need to use to make its life
# easier.
class BaseTable:

    def __init__(self, fileName, tableEntryId):
        self.fileName = fileName
        self.entryId = tableEntryId
        self.fileDict = dict()
        self.stopWordsPath = "C:\\Users\\Upal Hasan\\Desktop\\deepLearning\\stop_words.txt"
        self.stemmer = PorterStemmer()

        self.loadFileDictionary(fileName, tableEntryId)
        # keep set of keys for fast access later
        self.keySet = Set()
        for key in self.fileDict.keys():
            self.keySet.add(key)

        # load stop words
        self.stopWords = Set()
        fileHandle = open(self.stopWordsPath)
        stopWordList = [word.strip() for word in fileHandle.readlines() if word.strip() != ""]
        fileHandle.close()

        for word in stopWordList:
            self.stopWords.add(word)

    # the math to do perform the percentile for a given sorted and reversed
    # array and its corresponding length, along with the element to search for.
    def calculateNearestPercentile(self, array, elementToSearch, lengthOfArray, fieldName):
        elementIndex = float(array.index(elementToSearch))
        percentile = (lengthOfArray - elementIndex)/lengthOfArray
        percentile = round(10*percentile)*10

        return fieldName + "_" + str(percentile) + "_percentile"

    # loads the data for a given fileName and indexes it by the elementId
    def loadFileDictionary(self, fileName, elementId):
        fileHandle = open(fileName)

        # load the JSON file
        jsonList = [line.strip() for line in fileHandle.readlines()]
        jsonObject = [json.loads(jsonElement) for jsonElement in jsonList]
        fileHandle.close()

        for element in jsonObject:
            self.fileDict[element[elementId]] = element

    # this is the function that will take a field and try to return the
    # preprocessed feature for it. we first parse the field and based on
    # whether or not it's a compound field, we either return the first layer
    # of the dictionary or the second layer. we also preprocess the data by
    # calling the preprocessData() before returning to client. this function
    # mainly works on strings, so each derived class will need to handle its
    # numerical values differently.
    def generateFeatureVector(self, elementId, field):
        if elementId not in self.keySet:
            return []

        if not field:
            return []

        feature = ''
        feature_vector = []
        fieldSplit = field.split("|")
        if len(fieldSplit) == 1:
            if fieldSplit[0] in self.fileDict[elementId]:
               feature = self.preprocessData(
                            self.fileDict[elementId][fieldSplit[0]])
            else:
                print field + " not valid field."
        elif len(fieldSplit) == 2:
            if fieldSplit[0] in self.fileDict[elementId] \
                    and fieldSplit[1] in self.fileDict[elementId][fieldSplit[0]]:
                feature = self.preprocessData(
                        self.fileDict[elementId][fieldSplit[0]][fieldSplit[1]])
            else:
                print field + " not valid field."
        else:
            print field + " with " + len(fieldSplit) + " members not supported."

        if type(feature) == str:
            return feature_vector.append(feature)
        return feature

    # returns all the values of the dictionary
    def getData(self):
        return self.fileDict.values()

    # general function that will try to preprocess the data based on its type
    # for strings, call the preprocessText(). for lists, go through each element,
    # and call preprocessText.  Otherwise, it's a numerical data point, so we
    # must handle it differently.
    def preprocessData(self, data):
        if type(data) == str or type(data) == unicode:
            return self.preprocessText(data)
        elif type(data) == list:
            feature = []
            for element in data:
                feature.extend(self.preprocessText(element))
            return feature
        else:
            return data

    # this function is used to preprocess text data. we call gensim's
    # preprocess function and then for each word, we remove all newlines,
    # carriage returns, etc, and store only those words that aren't stop words.
    # of these stored words, we stem them and return the final list
    def preprocessText(self, text):
        preprocessedData = simple_preprocess(text)
        dataNoStop = [word.strip() for word in preprocessedData \
                            if not word in self.stopWords]

        dataStem = [self.stemmer.stem(word) for word in dataNoStop]
        return dataStem

# this class stores information about users
class UserTable(BaseTable):

    def __init__(self, fileName, tableEntryId):
        BaseTable.__init__(self, fileName, tableEntryId)

        self.userAverageStars = []
        self.userReviewCount = []
        self.userVotesCool = []
        self.userVotesFunny = []
        self.userVotesUseful = []
        self.fields = ["average_stars", "review_count", "name",
                        "votes|cool", "votes|funny", "votes|useful"]

    # this function will take a list of users and store a list for a given
    # numerical element of all the values that we've encountered for it. this
    # is used to eventually determine the percentile of a given number. we first
    # store all the numerical values, then sort them, and then reverse that
    # list. then we can index into this array and determine the percentile of
    # a given value.
    def convertNumericalFeatures(self, userIdList):
        for userId in userIdList:
            if userId not in self.keySet:
                continue

            self.userAverageStars.append(self.fileDict[userId]["average_stars"])
            self.userReviewCount.append(self.fileDict[userId]["review_count"])
            self.userVotesCool.append(self.fileDict[userId]["votes"]["cool"])
            self.userVotesFunny.append(self.fileDict[userId]["votes"]["funny"])
            self.userVotesUseful.append(self.fileDict[userId]["votes"]["useful"])

        self.userAverageStars = sorted(self.userAverageStars)[::-1]
        self.userReviewCount = sorted(self.userReviewCount)[::-1]
        self.userVotesCool = sorted(self.userVotesCool)[::-1]
        self.userVotesFunny = sorted(self.userVotesFunny)[::-1]
        self.userVotesUseful = sorted(self.userVotesUseful)[::-1]

        self.lengthUserAvgStars = len(self.userAverageStars)
        self.lengthUserReviewCount = len(self.userReviewCount)
        self.lengthUserVotesCool = len(self.userVotesCool)
        self.lengthUserVotesFunny = len(self.userVotesFunny)
        self.lengthUserVotesUseful = len(self.userVotesUseful)

    # this function will help with some of the preprocessing for the
    # user table data and gets invoked by generateFeatureVector.
    # for the relevant field, it will change it to a categorical field by
    # converting it into a percentile. if the field does not exist, it will
    # return a None object type.
    def getFeatureByField(self, field, element):
        if field == "review_count":
            return BaseTable.calculateNearestPercentile(self,
                self.userReviewCount, element["review_count"],
                self.lengthUserReviewCount, "review_count")
        elif field == "average_stars":
            return BaseTable.calculateNearestPercentile(self,
                self.userAverageStars, element["average_stars"],
                self.lengthUserAvgStars, "average_stars")
        elif field == "votes|cool":
            return BaseTable.calculateNearestPercentile(self,
                self.userVotesCool, element["votes"]["cool"],
                self.lengthUserVotesCool, "votes_cool")
        elif field == "votes|funny":
            return BaseTable.calculateNearestPercentile(self,
                self.userVotesFunny, element["votes"]["funny"],
                self.lengthUserVotesFunny, "votes_funny")
        elif field == "votes|useful":
            return BaseTable.calculateNearestPercentile(self,
                self.userVotesUseful, element["votes"]["useful"],
                self.lengthUserVotesUseful, "votes_useful")
        else:
            return None

    # this function will generate a feature vector for the user table.
    # it takes a list of fields that the client is interested in, and then
    # preprocesses the relevant data, and puts them into a vector to return
    # to the client
    # if no fields of interest are specified, we use all the fields.
    # TODO: it would be ideal if we could put this function in the base
    #       class and have it call the relevant getFeatureByField() depending
    #       on which class invoked it, but couldn't figure out how to do this
    #       in python. does polymorphism exist for it?
    def generateFeatureVector(self, userId, fieldsToInclude=[]):
        if userId not in self.keySet:
            return []

        if not fieldsToInclude:
            fieldsToInclude = self.fields;

        user_vector = []
        element = self.fileDict[userId]
        for field in fieldsToInclude:
            feature = BaseTable.generateFeatureVector(self, userId, field)
            if type(feature) == int or type(feature) == float:
                feature = self.getFeatureByField(field, element)
                if feature:
                    user_vector.append(feature)
                else:
                    print "feature doesn't exist: " + str(feature)
            elif type(feature) == list:
                user_vector.extend(feature)
            else:
                print "type not supported: " + str(feature)

        return user_vector

# this class stores information about the businesses
class BusinessTable(BaseTable):

    def __init__(self, fileName, tableEntryId):
        BaseTable.__init__(self, fileName, tableEntryId)

        self.businessReviewCount = []
        self.businessLat = []
        self.businessLong = []
        self.fields = ["review_count", "latitude", "longitude",
                        "full_address", "categories", "city", "name"]

    # this function will take a list of businesses and store a list for a given
    # numerical element of all the values that we've encountered for it. this
    # is used to eventually determine the percentile of a given number. we first
    # store all the numerical values, then sort them, and then reverse that
    # list. then we can index into this array and determine the percentile of
    # a given value.
    def convertNumericalFeatures(self, businessIdList):
        for businessId in businessIdList:
            if businessId not in self.keySet:
                continue

            self.businessReviewCount.append(self.fileDict[businessId]["review_count"])
            self.businessLat.append(self.fileDict[businessId]["latitude"])
            self.businessLong.append(self.fileDict[businessId]["longitude"])

        self.businessReviewCount = sorted(self.businessReviewCount)[::-1]
        self.businessLat = sorted(self.businessLat)[::-1]
        self.businessLong = sorted(self.businessLong)[::-1]

        self.lengthBusinessReviewCount = len(self.businessReviewCount)
        self.lengthBusinessLat = len(self.businessLat)
        self.lengthBusinessLong = len(self.businessLong)

    # this function will help with some of the preprocessing for the
    # business table data and gets invoked by generateFeatureVector.
    # for the relevant field, it will change it to a categorical field by
    # converting it into a percentile
    def getFeatureByField(self, field, element):
        if field == "review_count":
            return BaseTable.calculateNearestPercentile(self,
                self.businessReviewCount, element["review_count"],
                self.lengthBusinessReviewCount, "review_count")
        elif field == "latitude":
            return BaseTable.calculateNearestPercentile(self,
                self.businessLat, element["latitude"], self.lengthBusinessLat,
                "latitude")
        elif field == "longitude":
            return BaseTable.calculateNearestPercentile(self,
                self.businessLong, element["longitude"],
                self.lengthBusinessLong, "longitude")
        else:
            return None

    # this function will generate a feature vector for the business table.
    # it takes a list of fields that the client is interested in, and then
    # preprocesses the relevant data, and puts them into a vector to return
    # to the client
    # if no fields of interest are specified, we use all the fields.
    # TODO: it would be ideal if we could put this function in the base
    #       class and have it call the relevant getFeatureByField() depending
    #       on which class invoked it, but couldn't figure out how to do this
    #       in python. does polymorphism exist for it?
    def generateFeatureVector(self, businessId, fieldsToInclude=[]):
        if businessId not in self.keySet:
            return []

        if not fieldsToInclude:
            fieldsToInclude = self.fields;

        business_vector = []
        element = self.fileDict[businessId]
        for field in fieldsToInclude:
            feature = BaseTable.generateFeatureVector(self, businessId, field)
            if type(feature) == int or type(feature) == float:
                feature = self.getFeatureByField(field, element)
                if feature:
                    business_vector.append(feature)
                else:
                    print "feature doesn't exist: " + str(feature)
            elif type(feature) == list:
                business_vector.extend(feature)
            else:
                print "type not supported: " + str(feature)

        return business_vector

# this class stores information about Reviews
class ReviewTable(BaseTable):

    def __init__(self, fileName, tableEntryId):
        BaseTable.__init__(self, fileName, tableEntryId)

        self.reviewStars = []
        self.reviewVotesCool = []
        self.reviewVotesFunny = []
        self.reviewVotesUseful = []
        self.fields = ["stars", "votes|cool", "votes|funny",
                        "votes|useful", "text"]

    # this function will take a list of reviews and store a list for a given
    # numerical element of all the values that we've encountered for it. this
    # is used to eventually determine the percentile of a given number. we first
    # store all the numerical values, then sort them, and then reverse that
    # list. then we can index into this array and determine the percentile of
    # a given value.
    def convertNumericalFeatures(self, reviewIdList):
        for reviewId in reviewIdList:
            if reviewId not in self.keySet:
                continue

            self.reviewStars.append(self.fileDict[reviewId]["stars"])
            self.reviewVotesCool.append(self.fileDict[reviewId]["votes"]["cool"])
            self.reviewVotesFunny.append(self.fileDict[reviewId]["votes"]["funny"])
            self.reviewVotesUseful.append(self.fileDict[reviewId]["votes"]["useful"])

        self.reviewStars = sorted(self.reviewStars)[::-1]
        self.reviewVotesCool = sorted(self.reviewVotesCool)[::-1]
        self.reviewVotesFunny = sorted(self.reviewVotesFunny)[::-1]
        self.reviewVotesUseful = sorted(self.reviewVotesUseful)[::-1]

        self.lengthReviewStars = len(self.reviewStars)
        self.lengthReviewVotesCool = len(self.reviewVotesCool)
        self.lengthReviewVotesFunny = len(self.reviewVotesFunny)
        self.lengthReviewVotesUseful = len(self.reviewVotesUseful)

    # this function will help with some of the preprocessing for the
    # review table data and gets invoked by generateFeatureVector.
    # for the relevant field, it will change it to a categorical field either
    # by adding a flag next to it, or converting it into a percentile
    def getFeatureByField(self, field, element):
        if field == "stars":
            return str(element["stars"]) + "_stars"
        elif field == "votes|cool":
            return BaseTable.calculateNearestPercentile(self,
                self.reviewVotesCool, element["votes"]["cool"],
                self.lengthReviewVotesCool, "votes_cool")
        elif field == "votes|funny":
            return BaseTable.calculateNearestPercentile(self,
                self.reviewVotesFunny, element["votes"]["funny"],
                self.lengthReviewVotesFunny, "votes_funny")
        elif field == "votes|useful":
            return BaseTable.calculateNearestPercentile(self,
                self.reviewVotesUseful, element["votes"]["useful"],
                self.lengthReviewVotesUseful, "votes_useful")
        else:
            return None

    # this function will generate a feature vector for the review table.
    # it takes a list of fields that the client is interested in, and then
    # preprocesses the relevant data, and puts them into a vector to return
    # to the client
    # if no fields of interest are specified, we use all the fields.
    # TODO: it would be ideal if we could put this function in the base
    #       class and have it call the relevant getFeatureByField() depending
    #       on which class invoked it, but couldn't figure out how to do this
    #       in python. does polymorphism exist for it?
    def generateFeatureVector(self, reviewId, fieldsToInclude=[]):
        if reviewId not in self.keySet:
            return []

        if not fieldsToInclude:
            fieldsToInclude = self.fields;

        review_vector = []
        element = self.fileDict[reviewId]
        # iterate through all our fields, preprocess the data, and append
        # to our vector
        for field in fieldsToInclude:
            feature = BaseTable.generateFeatureVector(self, reviewId, field)
            if type(feature) == int or type(feature) == float:
                feature = self.getFeatureByField(field, element)
                if feature:
                    review_vector.append(feature)
                else:
                    print "feature doesn't exist: " + str(feature)
            elif type(feature) == list:
                review_vector.extend(feature)
            else:
                print "type not supported: " + str(feature)

        return review_vector

class FeatureGenerator:
    def __init__(self, review, user, business):
        self.reviewData = review
        self.userData = user
        self.businessData = business

    # function that will generate the feature vectors for the specified fields
    # from the client. we essentially will extract out the relevant pieces of
    # information from the data, stick them into a feature vector, and return
    # to client for processing.
    # there are three arrays required as input to this function, each input
    # specifying the list of fields for each table. if no list is specified,
    # we will use everything.
    def generateFeatureVectors(self, fieldsToProcess=[[],[],[]]):
        # subset fields to train on
        reviewTableFields = fieldsToProcess[0]
        userTableFields = fieldsToProcess[1]
        businessTableFields = fieldsToProcess[2]

        userIds = []
        businessIds = []
        reviewIds = []
        # first get the list of all the review ids based on the review data,
        # so we don't process users or businesses that aren't used
        for review in self.reviewData.getData():
            userIds.append(review["user_id"])
            businessIds.append(review["business_id"])
            reviewIds.append(review["review_id"])

        # convert the numbers to a feature that will preserve the semantic
        # meaning of it.
        time_start = time.time()
        self.reviewData.convertNumericalFeatures(reviewIds);
        self.businessData.convertNumericalFeatures(businessIds);
        self.userData.convertNumericalFeatures(userIds);
        time_end = time.time()

        print "convertFeatures time: " + str(time_end - time_start)

        time_start = time.time()
        features = []
        # now we want to actually generate the feature vector by calling each
        # table's corresponding function to return the fields of interest in
        # a list
        for review in self.reviewData.getData():
            userId = review["user_id"]
            businessId = review["business_id"]
            reviewId = review["review_id"]

            reviewFeature = self.reviewData.generateFeatureVector(
                                    reviewId, reviewTableFields)
            businessFeature = self.businessData.generateFeatureVector(
                                    businessId, businessTableFields)
            userFeature = self.userData.generateFeatureVector(
                                    userId, userTableFields)

            reviewFeature.extend(businessFeature)
            reviewFeature.extend(userFeature)

            # we keep appending the feature vectors for each review
            # to train later
            features.append(reviewFeature)
            #print reviewFeature
            #print "len: " + str(len(reviewFeature))

        time_end = time.time()
        print "generateFeatures time: " + str(time_end - time_start)
        return features

    # function that will take a weight-term matrix from word2vec, and take a
    # of sentences that we want to create a point cloud for, and generate
    # a vector of vectors for it.
    def generatePointCloud(self, model, sentences):
        pointCloud = []
        for feature in sentences:
            for word in feature:
                if word in model:
                    pointCloud.append(model[word])

        return pointCloud

if __name__ == '__main__':
    userFile = "C:\\Users\\Upal Hasan\\Desktop\\yelp_data\\yelp_phoenix_academic_dataset\\yelp_academic_dataset_user.json"
    businessFile = "C:\\Users\\Upal Hasan\\Desktop\\yelp_data\\yelp_phoenix_academic_dataset\\yelp_academic_dataset_business.json"
    reviewFile = "C:\\Users\\Upal Hasan\\Desktop\\yelp_data\\yelp_phoenix_academic_dataset\\yelp_academic_dataset_review.json"

    #reviewFile = "C:\\Users\\Upal Hasan\\Desktop\\yelp_data\\json.txt"
    #businessFile = "C:\\Users\\Upal Hasan\\Desktop\\yelp_data\\json1.txt"
    #userFile = "C:\\Users\\Upal Hasan\\Desktop\\yelp_data\\json2.txt"

    modelPath = "C:\\Users\\Upal Hasan\\Desktop\\deepLearning\\model\\model.out"
    wgtMatrixPath = "C:\\Users\\Upal Hasan\\Desktop\\deepLearning\\model\\wgt.out"

    # load up all the data from the three tables for faster access later
    time_start = time.time()
    userTable = UserTable(userFile, "user_id")
    businessTable = BusinessTable(businessFile, "business_id")
    reviewTable = ReviewTable(reviewFile, "review_id")
    time_end = time.time()

    print "tableLoad time: " + str(time_end - time_start)

    # create our features now
    generator = FeatureGenerator(reviewTable, userTable, businessTable)
    '''sentences = generator.generateFeatureVectors(
        [
            ["votes|funny","votes|useful","votes|cool","stars"],
            ["votes|funny","votes|useful","votes|cool","average_stars","review_count","name"],
            ["review_count","latitude","longitude","city","name"]
        ])
    '''
    # we are not providing any arguments, so we want to use all the data points
    sentences = generator.generateFeatureVectors()

    #print sentences

    print "Training model..."
    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    model.save(modelPath)
    model.save_word2vec_format(wgtMatrixPath)

    #pointCloud = generator.generatePointCloud(model, sentences)
    #print "len: " + str(len(pointCloud))

