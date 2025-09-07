import pandas 

# Module for data preprocessing 

def preProcessing(dataset: pandas.DataFrame):
    # Removing PlayerID, Gender and InGamePurchases columns
    dataset = dataset.drop(columns=["PlayerID", "Gender", "InGamePurchases"])

    # Transforming GameDifficulty, EngagementLevel into integer value ()
    dataset["GameDifficulty"] = dataset["GameDifficulty"].replace({"Easy": 0, "Medium": 1, "Hard": 2})

    dataset["EngagementLevel"] = dataset["EngagementLevel"].replace({"Low": 0, "Medium": 1, "High": 2})
        
    # Adding Location and GameGenre relative frequency (Freq. Encoding)
    locationFreq = dataset["Location"].value_counts() / len(dataset)
    gameGenreFreq = dataset["GameGenre"].value_counts() / len(dataset)

    dataset["LocationFreq"] = dataset["Location"].map(locationFreq)
    dataset["GameGenreFreq"] = dataset["GameGenre"].map(gameGenreFreq)

    return dataset
