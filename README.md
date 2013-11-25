DeepLearning
============

DeepLearning research project.

Here are some things that will need to be done if we move forward with this code:
* generalize for the other datasets 
  * find a general way to specify which fields for a given document vector has numerical values
  * figure out a general way to create the sentence vector to pass to gensim (since the length could vary based on which dataset you're looking at)
* figure out how to avoid loading all the numerical values into memory to sort and reverse (for the percentile calculation)
* play with the settings for gensim to determine the optimal dimensionality for each feature vector - current I'm using the defaults.
