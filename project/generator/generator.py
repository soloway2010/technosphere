import pandas as pd
import numpy as np

data = pd.read_csv("./data.csv", sep=',')

df = pd.DataFrame(columns=["Country", "Sex", "Age", "Marital status"])

unique_countries = np.unique(data.iloc[:, 1])

N = 100000

df["Country"] = np.random.choice(unique_countries, N)
df["Sex"] = np.random.choice(["Men", "Women"], N)
df["Age"] = np.random.choice(range(15, 100), N)

statuses = {}

for i in xrange(N):
	tmp = data[(data["Country"] == df["Country"][i]) & (data["Sex"] == df["Sex"][i])]
	age_group = df["Age"][i] / 5 + 1;
	if age_group > 14:
		age_group = 14
	
	probs = list(tmp.iloc[:, age_group] / 100)
	probs[0] += 1 - sum(probs)
	statuses[i] = np.random.choice(tmp["Marital status"], 1, p=probs)[0]

df["Marital status"] = list(statuses.values())

df.to_csv("./people.csv", sep=',')