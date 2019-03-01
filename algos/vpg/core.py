import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
	if shape is None:
		return (length, )
	return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
	return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
