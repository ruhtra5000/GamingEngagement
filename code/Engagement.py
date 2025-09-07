import pandas
from PreProcessing import preProcessing
from GraphPlotter import plotLikelihood, plotLikelihoodCategorical
from NaiveBayes import training

dataset = pandas.read_csv('online_gaming_behavior_dataset.csv')
dataset = preProcessing(dataset)

training(dataset)

#plotLikelihood(dataset, "AchievementsUnlocked")
#plotLikelihoodCategorical(dataset, "InGamePurchases")
