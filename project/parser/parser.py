import pandas as pd
import numpy as np

data_Raw = pd.read_csv("./data_Raw.csv", sep=',')

data = data_Raw.iloc[:, 0:18]

data = data.drop(["ISO code", "Non-standard age groups"], axis=1);

data = data[data.iloc[:, 1] == "Latest"]

data = data.drop(["Period", "Year"], axis=1)

data["Marital status"] = data["Marital status"].replace("Divorced/Separated", "Divorced")

data = data[(data.iloc[:, 2] == "Single") | (data.iloc[:, 2] == "Married") | (data.iloc[:, 2] == "Divorced")]

data = data[data.iloc[:, 1] != "Both Sexes"]

data = data.dropna()

data = data.reset_index(drop=True)

data.to_csv("./data.csv", sep=',')