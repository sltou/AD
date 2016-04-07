import numpy as np
from sklearn.mixture import GMM

def load_data(control_file, dementia_file):
	"""
	Loads feature vectors from file into matrices X_train and X_test
	"""
	X_control = []
	with open(control_file, 'r') as inf_control:
		for line in inf_control:
			features_str = line.split()
			features = map(float, features_str[1:])
			if len(features) == 0: continue  # in case there's empty lines in file
			X_control.append(features)

	X_dementia = []
	with open(dementia_file, 'r') as inf_dementia:
		for line in inf_dementia:
			features_str = line.split()
			features = map(float, features_str[1:])
			if len(features) == 0: continue
			X_dementia.append(features)

	return np.array(X_control),np.array(X_dementia)



if __name__ == "__main__":
	X_control,X_dementia = load_data("control_features_updated.txt", "dementia_features_updated.txt")

	# Add parameters as needed
	comp_range = [1,2,3,5]  # hyperparameter, try different number of mixture components
	covar_type = 'spherical'  # the only type I've learned (or remembered learning) tbh

	# TODO: actually split training and testing data
	# trying out the module first, then put in loop to test different param values and cross-validate
	model = GMM(n_components=1, covariance_type=covar_type)
	model.fit(X_control)

