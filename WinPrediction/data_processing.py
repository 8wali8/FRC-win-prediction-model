import pandas as pd

data_path = "Archimedes_Scouting_Info.csv"
# df = pd.read_csv(data_path)
df = pd.read_csv(data_path, skiprows=2)
# df2 = df.drop(labels="Archimedes Division", axis=0)
# df2 = df.drop(["Archimedes Division"], axis=1)
# print(df2)
print(df)
# print(df.iloc[1])
# print(df.iloc[1:])
