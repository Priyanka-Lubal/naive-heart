from sklearn.naive_bayes import GaussianNB
import numpy as np

Xtr = np.load("/home/yash/Desktop/heart/xtr.npy")
Ytr = np.load("/home/yash/Desktop/heart/Ytr.npy")

gnb = GaussianNB()
gnb.fit(Xtr, Ytr)
print(gnb.predict(Xtr))
