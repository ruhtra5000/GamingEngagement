import pandas
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

# Module for model training (using Naive Bayes)

def naiveBayesTraining(dataset: pandas.DataFrame, doRating = False):
    # Select features
    columns = ["Age", "SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", 
               "AchievementsUnlocked"] 
    target = "EngagementLevel"

    features = dataset[columns].values
    labels = dataset[target].values

    # Separate training and testing
    train, test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Chosing classifier (Naive Bayes)
    model = GaussianNB()

    # Training models
    model.fit(train, labels_train)

    # Rating
    if doRating:
        # Predict values
        predictedLabels = model.predict(test)
        
        print("\nMatriz de confusão:")
        print(confusion_matrix(labels_test, predictedLabels))

        print("\nRelatório de classificação:")
        print(classification_report(labels_test, predictedLabels))

    return model
