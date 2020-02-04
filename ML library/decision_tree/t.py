import pandas as pd
file = "data.csv"
df1 = pd.read_csv(file,encoding="gbk")
cols = list(df1.columns)[0:-1]
cols.append("为1的概率")
df = pd.DataFrame(None, columns=cols)
print(df)