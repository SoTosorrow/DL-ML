import numpy as np
from sklearn.decomposition import PCA
dataset = np.loadtxt("data.txt",delimiter="\t")
pca = PCA(n_components=3)
print(pca.fit_transform(dataset))
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=3)
print(svd.fit_transform(dataset))
