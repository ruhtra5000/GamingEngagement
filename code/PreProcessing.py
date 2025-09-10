import pandas 

# Module for data preprocessing 

def preProcessing(dataset: pandas.DataFrame):
    # Removing PlayerID, Gender and InGamePurchases columns
    dataset = dataset.drop(columns=["PlayerID", "Gender", "InGamePurchases"])

    # Transforming GameDifficulty and EngagementLevel into integer value (Label Encoding)
    dataset["GameDifficulty"] = dataset["GameDifficulty"].replace({"Easy": 0, "Medium": 1, "Hard": 2})
    
    dataset["EngagementLevel"] = dataset["EngagementLevel"].replace({"Low": 0, "Medium": 1, "High": 2})

    return dataset
