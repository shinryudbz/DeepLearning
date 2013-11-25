#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Upal Hasan
#
# Created:     23/11/2013
# Copyright:   (c) Upal Hasan 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os
from gensim.models.word2vec import *

def createModel():
    obamaDir = "C:\\Users\\Upal Hasan\\Desktop\\deepLearning\\obama"
    bushDir = "C:\\Users\\Upal Hasan\\Desktop\\deepLearning\\bush"
    modelPath = "C:\\Users\\Upal Hasan\\Desktop\\deepLearning\\model\\model.out"
    wgtMatrixPath = "C:\\Users\\Upal Hasan\\Desktop\\deepLearning\\model\\wgt.out"

    sentences = []
    os.chdir(obamaDir)
    for obamaPath in os.listdir(obamaDir):
        obamaFile = open(obamaPath)
        lines = [ map(str, line.strip().split(" "))
                    for line in obamaFile.readlines() if line.strip() != "" ]
        sentences.extend(lines)
        obamaFile.close()

    os.chdir(bushDir)
    for bushPath in os.listdir(bushDir):
        bushFile = open(bushPath)
        lines = [ map(str, line.strip().split(" "))
                    for line in bushFile.readlines() if line.strip() != "" ]
        sentences.extend(lines)
        bushFile.close()

    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    model.save(modelPath)
    model.save_word2vec_format(wgtMatrixPath)

if __name__ == '__main__':
    #createModel()

    modelPath = "C:\\Users\\Upal Hasan\\Desktop\\deepLearning\\model\\model.out"
    wgtMatrixPath = "C:\\Users\\Upal Hasan\\Desktop\\deepLearning\\model\\wgt.out"

    wgtMatrix = Word2Vec.load_word2vec_format(wgtMatrixPath)
    print wgtMatrix