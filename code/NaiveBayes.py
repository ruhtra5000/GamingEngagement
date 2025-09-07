import pandas
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import classification_report, confusion_matrix

def training(dataset: pandas.DataFrame):
    # Selecionar algumas features relevantes
    features = ["Age", "SessionsPerWeek", "AvgSessionDurationMinutes", 
                "PlayerLevel", "AchievementsUnlocked", "GameGenreFreq", "LocationFreq"]  
    target = "EngagementLevel"

    X = dataset[features].copy()
    y = dataset[target]

    # Separar em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Escolher o classificador
    # Se as features forem numéricas contínuas -> GaussianNB
    model = GaussianNB()

    # Treinar
    model.fit(X_train, y_train)

    # Avaliar
    y_pred = model.predict(X_test)

    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))
