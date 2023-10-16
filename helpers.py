

import copy as _copy
import numpy as np


## Helper functions

def chunker(seq, size):
	return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def dict_merge(*dict_list, **kwargs):
	"""Merge dictionary objects

	Contrary to :func:`dict.update`, perform a recursive merge (see examples).

	Parameters
	----------
	dict_list : list
		Input dictionaries to merge.
	kwargs : dict
		Merge parameters:

		* ``flag_copy`` [bool]: Control setting operation (copy or assignment).

	Returns
	-------
	dict
		Output dictionary.

	Examples
	--------
	>>> val = [1, 2, 3]
	>>> dict_0 = {1: val, 'b': {'a': 11, 'b': 12}, 'c': {'key': 'value'}}
	>>> dict_1 = {'b': {'b': 13}, 3: 'c', 'c': {}}
	>>> dict_out = merge(dict_0, None, dict_1)
	>>> dict_out_ref = {1: val, 'b': {'a': 11, 'b': 13}, 3: 'c', 'c': {}}
	>>> dict_out == dict_out_ref
	True
	>>> dict_out_c = merge(dict_0, None, dict_1, flag_copy=True)
	>>> dict_out_c == dict_out_ref
	True
	>>> val[0] = 12
	>>> dict_out == dict_out_ref
	True
	>>> dict_out_c == dict_out_ref
	False
	"""
	flag_copy = kwargs.get('flag_copy', False)
	dict_merged = {}
	for dict_t in dict_list:
		if dict_t is not None:
			for key, val in dict_t.items():
				if key not in dict_merged:
					if flag_copy:
						dict_merged[key] = _copy.deepcopy(val)
					else:
						dict_merged[key] = val
				else:
					if isinstance(val, dict) and \
							isinstance(dict_merged[key], dict) and \
							val:
						dict_merged[key] = merge(dict_merged[key], val)
					else:
						if flag_copy:
							dict_merged[key] = _copy.deepcopy(val)
						else:
							dict_merged[key] = val
	return dict_merged








## Helper funstions for diffusion models


def cos_beta_schedule(timesteps, offset_s=0.008, max_beta=0.999):
	"""cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
	beta = []
	alpha_bar = lambda t: np.cos((t + offset_s) / (1 + offset_s) * np.pi / 2) ** 2
	for i in range(timesteps):
		t1 = i / timesteps
		t2 = (i + 1) / timesteps
		beta.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
	return np.array(beta)


def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
	sigmoid = lambda t: 1 / (1 + np.exp(-t))
	beta = np.linspace(-6, 6, timesteps)
	return sigmoid(beta) * (beta_end - beta_start) + beta_start


def quadratic_beta_schedule(timesteps, beta_start, beta_end):
	return np.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def linear_beta_schedule(timesteps, beta_start, beta_end):
	return np.linspace(beta_start, beta_end, timesteps)


def get_beta_schedule(schedule_name, timesteps, **kwargs):
	"""
	Retrieve the beta scheduler function from the input string ``schedule_name``.

	Params
	---------

	:param schedule_name: String indicating the type of noise scheduling:
	``linear`` (or ``lin``), ``cosine`` (or ``cos``), ``quadratic`` (or ``quad``),
	and ``sigmoid`` (or ``sig``).
	:param timesteps: Number of timesteps T (e.g., 1000)
	:param kwargs: parameters for noise scheduling. beta_start, beta_end,
	offset_s, max_beta
	:return: beta schedule
	"""
	beta_start = kwargs.get('beta_start', None)
	beta_end = kwargs.get('beta_end', None)
	offset_s = kwargs.get('offset_s', None)
	max_beta = kwargs.get('max_beta', None)

	if schedule_name.lower() in ('lin', 'linear'):
		beta = linear_beta_schedule(timesteps, beta_start, beta_end)
	elif schedule_name.lower() in ('quad', 'quadratic'):
		beta = quadratic_beta_schedule(timesteps, beta_start, beta_end)
	elif schedule_name.lower() in ('sig', 'sigmoid'):
		beta = sigmoid_beta_schedule(timesteps, beta_start, beta_end)
	elif schedule_name.lower() in ('cos', 'cosine'):
		beta = cos_beta_schedule(timesteps, offset_s=offset_s, max_beta=max_beta)
	else:
		raise NotImplementedError('Schedule name ({}) not recognized or '
		                          'not implemented.'.format(schedule_name))
	return beta


