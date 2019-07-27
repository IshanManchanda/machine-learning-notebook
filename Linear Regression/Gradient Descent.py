import numpy as np
from matplotlib import pyplot as plt

from utils import generate_data


def gradient_descent(x, y, alpha, features, cases, iterations, epsilon):
	thetas = np.ones(shape=features + 1)
	x_t = x.transpose()
	precision = int(abs(np.log10(epsilon))) + 1
	costs = []
	prev_cost = np.inf

	for i in range(iterations):
		difference = np.dot(x, thetas) - y
		cost = np.sum(difference ** 2) / (2 * cases)
		costs.append(cost)
		print(("Iteration %d | Cost: {:.%df}" % (i, precision)).format(cost))

		if cost < epsilon:
			print("Cost < epsilon. Exiting.")
			break

		if prev_cost - cost < epsilon:
			print("Change in cost < epsilon. Exiting.")
			break

		gradient = np.dot(x_t, difference) / cases
		thetas = thetas - alpha * gradient
	else:
		print("Iterations completed.")
	return thetas, costs


def plot_costs(costs):
	plt.figure(dpi=400)
	plt.plot(costs)
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.show()


def main():
	features, cases, variance = 5, 10000, 0
	alpha, iterations, epsilon = 0.000000005, 100000, 0.000001

	x, y, generated_thetas = generate_data(features, cases, variance)
	calculated_thetas, costs = gradient_descent(
		x, y, alpha, features, cases, iterations, epsilon
	)

	plot_costs(costs)
	print(f'Generated thetas: {generated_thetas}')
	print(f'Calculated thetas: {calculated_thetas}')


if __name__ == '__main__':
	main()
