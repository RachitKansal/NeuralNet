#!/usr/bin/env python3
import numpy as np

def normalize(X):
	X = X.astype(float)
	for i in range(len(X[0])):
		X[:, [i]] = X[:, [i]] - X[:, [i]].mean()
		X[:, [i]] = X[:, [i]] - X[:, [i]].std()
	return X