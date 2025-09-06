import pandas 

# Module for data preprocessing 

def preProcessing():
    # Loading dataset
    dataset = pandas.read_csv('online_gaming_behavior_dataset.csv')

    # Removing PlayerID column
    dataset = dataset.drop(columns=["PlayerID"])

    # Transforming Gender, GameDifficulty, EngagementLevel into integer value
    dataset = dataset.rename(columns={"Gender": "Gender_Male"})
    dataset["Gender_Male"] = dataset["Gender_Male"].replace({"Male": 1, "Female": 0})

    dataset["GameDifficulty"] = dataset["GameDifficulty"].replace({"Easy": 0, "Medium": 1, "Hard": 2})

    dataset["EngagementLevel"] = dataset["EngagementLevel"].replace({"Low": 0, "Medium": 1, "High": 2})
        
    print(dataset.dtypes) 

    print(dataset.describe())

    print(f"\nLocation values: {dataset["Location"].unique()}")
    print(f"\nGame genre values: {dataset["GameGenre"].unique()}")

    return dataset
