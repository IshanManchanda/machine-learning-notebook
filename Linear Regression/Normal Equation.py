import numpy as np

from utils import generate_data


def normal_equation(x, y):
	x_t = x.transpose()
	xx_t = x_t.dot(x)

	try:
		inv_xx_t = np.linalg.inv(xx_t)
	except np.linalg.LinAlgError:
		print('Un-invertible Matrix')
		return None

	return inv_xx_t.dot(x_t).dot(y)


def main():
	features, cases, variance = 5, 10000, 0
	x, y, thetas = generate_data(features, cases, variance)
	print(f'Generated thetas: {thetas}')

	thetas = normal_equation(x, y)
	print(f'Calculated thetas: {thetas}')


if __name__ == '__main__':
	main()
