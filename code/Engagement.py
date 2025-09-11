import pandas
from GraphPlotter import plotLikelihood, plotLikelihoodCategorical
from PreProcessing import preProcessing
from NaiveBayes import naiveBayesTraining
from sklearn.naive_bayes import GaussianNB
from Interface import showWebInterface

# Open dataset and preprocessing it
dataset = pandas.read_csv('online_gaming_behavior_dataset.csv')
dataset = preProcessing(dataset)

# Plot likelihood graphs
#plotLikelihood(dataset, "AchievementsUnlocked")
#plotLikelihoodCategorical(dataset, "InGamePurchases")

# Training model
trainedModel = naiveBayesTraining(dataset)

# Show interface
showWebInterface(trainedModel)
