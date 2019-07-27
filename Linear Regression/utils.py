import random

import numpy as np


def generate_data(features, cases, variance):
	x = np.zeros(shape=(cases, features + 1))
	y = np.zeros(shape=cases)
	thetas = np.zeros(shape=features + 1)

	for i in range(features):
		thetas[i] = random.uniform(0, i ** 2 + 1)

	for i in range(cases):
		x[i][0] = 1
		for j in range(features):
			x[i][j + 1] = random.uniform(0, i * j + 1)
		y[i] = np.sum(np.dot(x[i], thetas)) + random.uniform(-1, 1) * variance
	return x, y, thetas
