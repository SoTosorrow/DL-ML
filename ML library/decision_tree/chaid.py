from sklearn.feature_selection import SelectKBest,chi2
import pandas as pd

file = "data.csv"

df = pd.read_csv(file,encoding="gbk")
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
new_data = SelectKBest(chi2,k=1).fit_transform(X,y)
print(new_data)
