import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

DATA_FILE = "../data/train_data_final_50k.csv"
DATA_SIZE = 50000


def load_data(num_data_points=50000):
	skip_rows = DATA_SIZE - num_data_points

	# load the data and split it into training/testing sets
	data = np.loadtxt(DATA_FILE, delimiter=',', skiprows=skip_rows, usecols=range(4,622))
	labels = np.loadtxt(DATA_FILE, delimiter=',', skiprows=skip_rows, usecols=622)
	X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.7, stratify=labels)

	# feature scale the data
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

	# reduce dimensionality of the data
	svd = TruncatedSVD(n_components=90)
	X_train_svd = svd.fit_transform(X_train)
	X_test_svd = svd.transform(X_test)

	return (X_train_svd, X_test_svd, y_train, y_test)
