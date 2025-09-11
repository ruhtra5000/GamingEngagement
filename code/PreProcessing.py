import pandas 

# Module for data preprocessing 

def preProcessing(dataset: pandas.DataFrame):
    # Removing columns
    dataset = dataset.drop(columns=[
        "PlayerID", 
        "Gender", 
        "InGamePurchases", 
        "GameGenre", 
        "Location",
        "GameDifficulty"
    ])

    # Transforming EngagementLevel into integer value (Label Encoding)
    dataset["EngagementLevel"] = dataset["EngagementLevel"].replace({"Low": 0, "Medium": 1, "High": 2})

    return dataset
