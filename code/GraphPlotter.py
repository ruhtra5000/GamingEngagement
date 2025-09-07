import pandas
import seaborn as sb
import matplotlib.pyplot as plt

# Module for plotting graphs (mainly likelihood graphs)

# Graph for continuous feature
def plotLikelihood(dataset, feature, target="EngagementLevel"):
    plt.figure(figsize=(8, 5))

    for cls in dataset[target].unique():
        subset = dataset[dataset[target] == cls]
        sb.kdeplot(
            subset[feature], 
            fill=True, 
            alpha=0.35, 
            label=f"{target}={cls}"
        )
    
    plt.title(f"Distribuição de {feature} para {target}")
    plt.xlabel(feature)
    plt.ylabel(f"P({target} | {feature})")
   
    plt.show()

#Graph for categorical feature
def plotLikelihoodCategorical(dataset, feature, target="EngagementLevel"):
    plt.figure(figsize=(8, 5))

    probTable = pandas.crosstab(
        dataset[feature], 
        dataset[target],
        normalize="columns"
    )

    probTable.plot(kind="bar", alpha=0.7)
    plt.title(f"Distribuição de {feature} para {target}")
    plt.xlabel(feature)
    plt.ylabel(f"P({target} | {feature})")
    plt.legend(title=target)

    plt.show()