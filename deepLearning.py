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

# debug flag
debug = False

if __name__ == '__main__':
    fileHandle = open("C:\\Users\\Upal Hasan\\Desktop\\yelp_data\\yelp_phoenix_academic_dataset\\yelp_academic_dataset_user.json")
    modelPath = "C:\\Users\\Upal Hasan\\Desktop\\deepLearning\\model\\model.out"
    wgtMatrixPath = "C:\\Users\\Upal Hasan\\Desktop\\deepLearning\\model\\wgt.out"

    # load the JSON file
    jsonList = [line.strip() for line in fileHandle.readlines()]
    jsonObject = [json.loads(jsonElement) for jsonElement in jsonList]
    fileHandle.close()

    # preprocess by converting numerical to percentiles
    averageStars = []
    reviewCount = []
    votesCool = []
    votesFunny = []
    votesUseful = []
    for element in jsonObject:
        averageStars.append(element["average_stars"])
        reviewCount.append(element["review_count"])
        votesCool.append(element["votes"]["cool"])
        votesFunny.append(element["votes"]["funny"])
        votesUseful.append(element["votes"]["useful"])

    # now sort and reverse
    # TODO: not a permanent solution and will need to be optimized to avoid
    #       getting long lists into memory
    averageStars = sorted(averageStars)[::-1]
    reviewCount = sorted(reviewCount)[::-1]
    votesCool = sorted(votesCool)[::-1]
    votesFunny = sorted(votesFunny)[::-1]
    votesUseful = sorted(votesUseful)[::-1]

    # help with debugging
    if debug:
        print "----------printing sorted numerical arrays"
        print averageStars
        print reviewCount
        print votesCool
        print votesFunny
        print votesUseful

    # store lengths of list for percentile calculation
    lengthAvgStars = len(averageStars)
    lengthReviewCount = len(reviewCount)
    lengthVotesCool = len(votesCool)
    lengthVotesFunny = len(votesFunny)
    lengthVotesUseful = len(votesUseful)

    # create the set of "words" here for gensium
    sentences = []
    for element in jsonObject:
        # store the words for a document (i.e. a user record)
        user_vector = [
                        str(100*(lengthAvgStars - averageStars.index(element["average_stars"]))/lengthAvgStars),
                        element["name"],
                        str(100*(lengthReviewCount - reviewCount.index(element["review_count"]))/lengthReviewCount),
                        element["type"],
                        element["user_id"],
                        str(100*(lengthVotesCool - votesCool.index(element["votes"]["cool"]))/lengthVotesCool),
                        str(100*(lengthVotesFunny - votesFunny.index(element["votes"]["funny"]))/lengthVotesFunny),
                        str(100*(lengthVotesUseful - votesUseful.index(element["votes"]["useful"]))/lengthVotesUseful)
                      ]
        # append it to running list of words for entire corpus (i.e. all user records)
        sentences.append(user_vector)

        if debug:
            print "-------printing user vector"
            print user_vector

    if debug:
        print "--------printing total list of words to gensium"
        print sentences

    # now calculate the feature vectors with gensium and save the model
    # and the weight matrix
    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    model.save(modelPath)
    model.save_word2vec_format(wgtMatrixPath)

''' user
{u'average_stars': 5.0,
 u'name': u'Jim',
 u'review_count': 6,
 u'type': u'user',
 u'user_id': u'CR2y7yEm4X035ZMzrTtN9Q',
 u'votes': {u'cool': 0, u'funny': 0, u'useful': 7}}
'''

''' reviews
{u'business_id': u'9yKzy9PApeiPPOUJEtnvkg',
 u'date': u'2011-01-26',
 u'review_id': u'fWKvX83p0-ka4JS3dc6E5A',
 u'stars': 5,
 u'text': u'My wife took me here on my birthday for breakfast and it was excellent.
 The weather was perfect which made sitting outside overlooking their grounds an absolute pleasure.
 Our waitress was excellent and our food arrived quickly on the semi-busy Saturday morning.
 It looked like the place fills up pretty quickly so the earlier you get here the better.\n\n
 Do yourself a favor and get their Bloody Mary.  It was phenomenal and simply the best I\'ve ever
 had.  I\'m pretty sure they only use ingredients from their garden and blend them fresh when you
 order it.  It was amazing.\n\nWhile EVERYTHING on the menu looks excellent, I had the white
 truffle scrambled eggs vegetable skillet and it was tasty and delicious.  It came with 2 pieces
 of their griddled bread with was amazing and it absolutely made the meal complete.  It was the
 best "toast" I\'ve ever had.\n\nAnyway, I can\'t wait to go back!',
 u'type': u'review',
 u'user_id': u'rLtl8ZkDX5vH5nAx9C3q5Q',
 u'votes': {u'cool': 2, u'funny': 0, u'useful': 5}}
'''

''' business
{u'business_id': u'rncjoVoEFUJGCUoC1JgnUA',
 u'categories': [u'Accountants',
                 u'Professional Services',
                 u'Tax Services',
                 u'Financial Services'],
 u'city': u'Peoria',
 u'full_address': u'8466 W Peoria Ave\nSte 6\nPeoria, AZ 85345',
 u'latitude': 33.581867,
 u'longitude': -112.241596,
 u'name': u'Peoria Income Tax Service',
 u'neighborhoods': [],
 u'open': True,
 u'review_count': 3,
 u'stars': 5.0,
 u'state': u'AZ',
 u'type': u'business'}
'''

''' checkin
{u'business_id': u'oRqBAYtcBYZHXA7G8FlPaA',
 u'checkin_info': {u'0-0': 3,
                   u'0-5': 1,
                   u'1-6': 2,
                   u'10-6': 1,
                   u'11-2': 1,
                   u'12-6': 1,
                   u'13-6': 1,
                   u'14-6': 1,
                   u'17-1': 1,
                   u'17-4': 1,
                   u'18-0': 1,
                   u'18-5': 4,
                   u'19-2': 1,
                   u'19-4': 1,
                   u'2-5': 3,
                   u'2-6': 2,
                   u'20-3': 1,
                   u'23-0': 1,
                   u'3-5': 1,
                   u'3-6': 1},
 u'type': u'checkin'}
'''