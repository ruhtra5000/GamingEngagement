import pandas
from PreProcessing import preProcessing
from GraphPlotter import plotLikelihood, plotLikelihoodCategorical

dataset = pandas.read_csv('online_gaming_behavior_dataset.csv')
#dataset = preProcessing(dataset)

#plotLikelihood(dataset, "AchievementsUnlocked")
plotLikelihoodCategorical(dataset, "GameDifficulty")
