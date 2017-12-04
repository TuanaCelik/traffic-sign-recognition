import numpy as np


def batch_generator(dataset, group, batch_size=100):

	idx = 0
	dataset_size = dataset['y_{0:s}'.format(group)].shape[0]
	indices = range(dataset_size)
	np.random.shuffle(indices)
	while idx < dataset_size:
		chunk = slice(idx, idx+batch_size)
		chunk = indices[chunk]
		chunk = sorted(chunk)
		idx = idx + batch_size
		yield dataset['X_{0:s}'.format(group)][chunk], dataset['y_{0:s}'.format(group)][chunk]