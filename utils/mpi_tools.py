from mpi4py import MPI
import os, subprocess, sys
import numpy as np

def mpi_statistics_scalar(x, with_min_and_max=False):

	x = np.array(x, dtype=np.float32)
	global_sum, global_n = mpi_sum([np.sum(x), len(x)])