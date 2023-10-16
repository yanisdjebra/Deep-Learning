import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Conv2DTranspose, concatenate, Layer
import tensorflow.keras.layers as nn
import inspect

import numpy as np
import helpers as hlp

# Clear previously registered custom objects from this module
del_key = []
for key in tf.keras.utils.get_custom_objects().keys():
	if 'CustomLayers' in key:
		del_key.append(key)
for del_key_i in del_key:
	tf.keras.utils.get_custom_objects().pop(del_key_i)


## Basic Layers and Blocks

def Upsample(dim):
	return nn.Conv2DTranspose(filters=dim, kernel_size=4, strides=2, padding='SAME')


def Downsample(dim):
	return nn.Conv2D(filters=dim, kernel_size=4, strides=2, padding='SAME')


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class SinusoidalPosEmb(Layer):
	def __init__(self, dim, max_positions=10000.):
		super(SinusoidalPosEmb, self).__init__()
		self.dim = dim
		self.max_positions = max_positions

	def call(self, x, training=True):
		x = tf.cast(x, tf.float64)
		half_dim = self.dim // 2
		emb = tf.cast(tf.math.log(self.max_positions), dtype=tf.float64) / (half_dim - 1)
		emb = tf.exp(tf.range(half_dim, dtype=tf.float64) * -emb)
		emb = x[:, None] * emb[None, :]

		emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

		return emb

	def get_config(self):
		config = super().get_config()
		config.update({
			"dim":           self.dim,
			"max_positions": self.max_positions,
		})
		return config


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class SiLU(Layer):
	def __init__(self):
		super(SiLU, self).__init__()

	def call(self, x, training=True):
		return x * tf.nn.sigmoid(x)


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
def gelu_func(x, approximate=False):
	if approximate:
		coeff = tf.cast(0.044715, x.dtype)
		return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
	else:
		return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class GELU(Layer):
	def __init__(self, approximate=False):
		super(GELU, self).__init__()
		self.approximate = approximate

	def call(self, x, training=True):
		return gelu_func(x, self.approximate)

	def get_config(self):
		config = super().get_config()
		config.update({
			"approximate": self.approximate,
		})
		return config


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class NormalizationLayer(tf.keras.layers.Layer):
	"""
	Pre-processing layer that normalizes or standardizes the input.
	Pass named arguments ``min`` and ``max`` to scale the input from 0 to 1 (normalize).
	Pass named arguments ``mean`` and ``var`` (or ``std``) to scale the distribution
	of the input data to a normal gaussian (standardize).
	flag ``invert=True`` indicates that the layer should undo the normalization.
	"""
	def __init__(self, min=None, max=None, mean=None, var=None,
	             std=None, invert=False, **kwargs):
		super().__init__(**kwargs)
		self.min = min
		self.max = max
		self.mean = mean
		self.var = var
		self.std = std
		self.invert = invert

		if (min is not None) and (max is not None):
			self.var_sub = min
			self.var_div = (max - min)
			self.mode = 'minmax'
		elif mean is not None:
			self.var_sub = mean
			self.var_div = np.sqrt(var) if var is not None else std
			self.mode = 'std'
		else:
			raise ValueError('``min`` and ``max`` or ``mean`` and '
			                 '``var`` (or ``std``) must be passed as '
			                 'input arguments')

	def call(self, inputs, **kwargs):
		if type(inputs) is list:
			raise NotImplementedError('')
		if not self.invert:
			outputs = (inputs - self.var_sub) / self.var_div
		else:
			outputs = inputs * self.var_div + self.var_sub

		return outputs

	def get_config(self):
		config = super().get_config()
		config.update({
			'min':    self.min,
			'max':    self.max,
			'mean':   self.mean,
			'var':    self.var,
			'std':    self.std,
			'invert': self.invert,
		})
		return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class Identity(Layer):
	def __init__(self):
		super(Identity, self).__init__()

	def call(self, x, training=True):
		return tf.identity(x)


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class Residual(Layer):
	def __init__(self, fn):
		super(Residual, self).__init__()
		self.fn = fn

	def call(self, x, training=True):
		return self.fn(x, training=training) + x

	def get_config(self):
		config = super().get_config()
		config.update({
			"fn": self.fn,
		})
		return config


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class MLP(Layer):
	def __init__(self, hidden_dim, **kwargs):
		super(MLP, self).__init__(**kwargs)
		self.net = Sequential([
			nn.Flatten(),
			nn.Dense(units=hidden_dim),
			GELU(),
			LayerNorm(hidden_dim),
			nn.Dense(units=hidden_dim),
			GELU(),
			LayerNorm(hidden_dim),
			nn.Dense(units=hidden_dim),
		])

	def call(self, x, training=True):
		return self.net(x, training=training)


## Advanced Layers and Blocks


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class Encoder_v3(Layer):
	"""Maps Inputs to latent variables z."""

	def __init__(self, enc_size, latent_dim=16, activation='relu',
	             flag_skip=False, **kwargs):
		super().__init__()

		self.enc_size = enc_size
		self.n_layers = len(enc_size)
		self.activation = activation
		self.latent_dim = latent_dim
		self.flag_skip = flag_skip
		self.encoder = {}
		self.dense_z = None
		self.kwargs = kwargs

	def build(self, input_shape):
		for ee in range(self.n_layers):
			self.encoder['layer_' + format(ee)] = nn.Dense(self.enc_size[ee],
			                                                   activation=self.activation,
			                                                   name='hidden_encoder_layer_{}'.format(ee),
			                                                   **self.kwargs)
		self.dense_z = nn.Dense(self.latent_dim, name='dense_z', **self.kwargs)

	def call(self, inputs, training=False):
		x = inputs
		h = []
		for ee in range(self.n_layers):
			x = self.encoder['layer_' + format(ee)](x, training=training)
			h.append(x)

		z = self.dense_z(x, training=training)
		return z, h

	def get_config(self):
		base_config = super().get_config()
		base_config.update({
			'latent_dim': self.latent_dim,
			'enc_size':   self.enc_size,
			'activation': self.activation,
			'flag_skip':  self.flag_skip
		})
		return {**base_config, **self.kwargs}

@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class Encoder_v3_noskip(Encoder_v3):
	"""Maps Inputs to latent variables z."""

	def call(self, inputs, training=False):
		x = inputs
		for ee in range(self.n_layers):
			x = self.encoder['layer_' + format(ee)](x, training=training)

		z = self.dense_z(x, training=training)
		return z


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class Decoder_v3(Layer):
	"""Converts z, the encoded latent vector, back into the original input space."""

	def __init__(self, dec_size, output_dim, activation='relu',
	             flag_skip=False, **kwargs):
		super().__init__()
		self.dec_size = dec_size
		self.n_layers = len(dec_size)
		self.output_dim = output_dim
		self.activation = activation
		self.flag_skip = flag_skip
		self.decoder = {}
		self.dense_output = None
		self.kwargs = kwargs

	def build(self, input_shape=None):
		for ee in range(self.n_layers):
			self.decoder['layer_' + format(ee)] = nn.Dense(self.dec_size[ee],
			                                                   activation=self.activation,
			                                                   name='hidden_decoder_layer_{}'.format(ee),
			                                                   **self.kwargs)
		self.dense_output = nn.Dense(self.output_dim, activation=self.activation,
		                                 name='Decoder_output_layer', **self.kwargs)

	def call(self, inputs, training=False):
		if len(inputs) > 1:
			x, h = inputs
		else:
			x = inputs
			h = None

		for ee in range(self.n_layers):
			x = self.decoder['layer_' + format(ee)](x, training=training)
			# if (ee % 2) == 0:
			# 	x = nn.Dropout(rate=0.1)(x)
			if self.flag_skip:
				x = nn.Add(name='skip_connection_{}'.format(ee))([x, h.pop(-1)])

		output = self.dense_output(x, training=training)
		return output

	def get_config(self):
		base_config = super().get_config()
		base_config.update({
			'output_dim': self.output_dim,
			'dec_size':   self.dec_size,
			'activation': self.activation,
			'flag_skip':  self.flag_skip
		})
		return {**base_config}

	@classmethod
	def from_config(cls, config):
		return cls(**config, **self.kwargs)


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class ConvBlock(Layer):
	"""Convolution block"""

	def __init__(self, ndim, units, num_convs=1, kernel_size=3, conv_args=None,
	             norm_list=None, norm_type='layer', activation='relu',
	             l2_reg=None, flag_res=False, *args, **kwargs):
		"""Constructor

		Parameters
		----------
		ndim : int
			Input dimensions (excluding batch and channel dimensions).
		units : int
			Number of convolution units.
		num_conv_per_block : int
			Number of consecutive convolution layers.
		kernel_size : int or list(int)
			Convolution kernel size.
		conv_args : dict
			Additional parameters passed to convolutional layers.
		norm_list : str
			Batch normalization (disabled if None, ``pre`` or ``post``
			activation).
		norm_type : str
			Normalization type (``batch`` or ``layer``).
		activation : str
			Activation function.
		l2_reg : float
			Kernel regularization.
		flag_res : bool
			When True, use ResNet-like skip connection (additive) around each
			block.
		"""
		# Attributes
		self.units = units
		self.ndim = ndim
		self.num_convs = num_convs
		self.conv_args = conv_args or {}
		if norm_list is None:
			self.norm_list = []
		elif isinstance(norm_list, str):
			self.norm_list = [s.strip() for s in norm_list.split(',')]
		elif isinstance(norm_list, (list, tuple)):
			self.norm_list = norm_list
		self.norm_type = norm_type
		self.activation = activation
		self.flag_res = flag_res

		if isinstance(kernel_size, (list, tuple)):
			kernel_size_d = kernel_size
		else:
			kernel_size_d = (kernel_size,) * ndim
		self.kernel_size = kernel_size_d
		self.l2_reg = l2_reg

		# Call parent constructor
		super().__init__(*args, **kwargs)

	def build(self, input_shape):
		"""Create layers for convolution block"""
		# Helper functions
		f_conv = getattr(nn, 'Conv{}D'.format(self.ndim))
		if self.norm_type == 'batch':
			f_norm = nn.BatchNormalization
		elif self.norm_type == 'layer':
			f_norm = nn.LayerNormalization
		else:
			raise NotImplementedError('``norm_type`` input argument not recognized '
			                          '(given {})'.format(self.norm_type))
		# Weight regularizer
		conv_args = self.conv_args.copy()
		if self.l2_reg is not None:
			conv_args['kernel_regularizer'] = tf.keras.regularizers.L2(
				self.l2_reg)
		# Define layers
		self.conv_l = [f_conv(self.units, kernel_size=self.kernel_size,
		                      **conv_args)
		               for c in range(self.num_convs)]
		if self.flag_res:
			self.res_conv = f_conv(self.units, kernel_size=1, **conv_args)
			self.res_add = nn.Add()
		self.activation_l = nn.Activation(self.activation)
		# Normalization
		if 'pre' in self.norm_list:
			self.norm_pre_l = [f_norm() for c in range(self.num_convs)]
		if 'post' in self.norm_list:
			self.norm_post_l = [f_norm() for c in range(self.num_convs)]

	def call(self, x, training=False):
		"""Layer call"""
		x_s = x
		for fi in range(self.num_convs):
			x = self.conv_l[fi](x, training=training)
			if 'pre' in self.norm_list:
				x = self.norm_pre_l[fi](x, training=training)
			if self.flag_res and fi == self.num_convs - 1:
				x = self.res_add([x, self.res_conv(x_s)])
			x = self.activation_l(x)
			if 'post' in self.norm_list:
				x = self.norm_post_l[fi](x, training=training)
		return x

	def get_config(self):
		"""Serialize object"""
		base_config = super().get_config()
		config = {'units':       self.units,
		          'num_convs':   self.num_convs,
		          'kernel_size': self.kernel_size,
		          'conv_args':   self.conv_args,
		          'ndim':        self.ndim,
		          'norm_list':   self.norm_list,
		          'norm_type':   self.norm_type,
		          'activation':  self.activation,
		          'l2_reg':      self.l2_reg,
		          'flag_res':    self.flag_res}
		return {**base_config, **config}

	@classmethod
	def from_config(cls, config):
		"""Create object from configuration"""
		return cls(**config)


## Complex Blocks and Layers using custom layers


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class UnetConditional(Layer):
	"""Conditional U-Net implemented as layer for an end-to-end training."""

	def __init__(self,  num_channels_out=1, num_filt_start=64,
	             pool_size=2, depth=5, block_type='conv', block_params=None,
	             skip_conn_type='concat', skip_conn_op=None,
	             skip_conn_post_op=None, dropout=None,
	             final_activation=None, cond_params=None,
	             sin_emb_dim=None, normalize_feature_dict=None,
	             normalize_label_dict=None, **kwargs):
		super(UnetConditional, self).__init__()

		self.num_channels_out = num_channels_out
		self.num_filt_start = num_filt_start
		self.pool_size = pool_size
		self.depth = depth
		self.block_type = block_type
		self.block_params = block_params
		self.skip_conn_type = skip_conn_type
		self.skip_conn_op = skip_conn_op
		self.skip_conn_post_op = skip_conn_post_op
		self.dropout = dropout
		self.final_activation = final_activation
		self.sin_emb_dim = sin_emb_dim
		self.sinusoidal_cond_mlp = False if sin_emb_dim in (None, 0) else True
		self.cond_params = cond_params or {'network_name': 'encoder',
		                                   'flag_flatten_input': True,
		                                   'network_kwargs': {'enc_size': [256, 128, 64],
		                                                      'latent_dim': 32}
		                                   }
		self.normalize_feature_dict = normalize_feature_dict or {}
		self.normalize_label_dict = normalize_label_dict or {}
		# self.n_condition = n_condition  # number of conditions to embed (e.g., time and digit)

		self.conv_args = {'padding': 'same'}
		self.ndim = None
		self.im_size = None

		# Create layers
		self.downs = []
		self.ups = []

		# Conditional embedding
		self.cond_mlp_down = []
		self.cond_mlp_up = []
		self.normalize_layer = {}
		self.denormalize_layer = {}

	def build(self, input_shape):

		# Build Normalization layers (create Normalization layers if normalize_*_dict is not None)
		for name_i, normalize_dict in zip(['feature', 'label'],
		                                  [self.normalize_feature_dict,
		                                   self.normalize_label_dict]):
			lower_keys = list(map(str.lower, normalize_dict.keys()))
			self.normalize_layer[name_i] = None
			self.denormalize_layer[name_i] = None
			if ('mean' in lower_keys) and ('var' in lower_keys or 'std' in lower_keys):
				normalize_kwargs = {
					'mean': np.array(normalize_dict.get('mean')),
					'var': np.array(normalize_dict.get('var', None)) if 'var' in lower_keys \
					else (np.array(normalize_dict.get('std', None)) ** 2)
				}
			elif ('min' in lower_keys) and ('max' in lower_keys):
				normalize_kwargs = {
					'min': np.array(normalize_dict.get('min')),
					'max': np.array(normalize_dict.get('max'))
				}
			else:
				raise ValueError('Normalization method for {} not recognized. Normalization dict '
				                 'should have named variables ``min`` and ``max`` or ``mean`` and '
				                 '``var`` (or ``std``). Given {}'.format(name_i, normalize_dict.keys()))

			# Create a Normalization layer and set its internal state using passed mean and variance
			self.normalize_layer[name_i] = NormalizationLayer(**normalize_kwargs)
			self.denormalize_layer[name_i] = NormalizationLayer(invert=True, **normalize_kwargs)

		# Get shapes
		ndim = len(input_shape) - 2
		self.ndim = ndim
		if ndim == 1:
			im_size = (input_shape[1],)
		elif ndim == 2:
			im_size = (input_shape[1], input_shape[2])
		elif ndim == 3:
			im_size = (input_shape[1], input_shape[2], input_shape[3])
		else:
			raise NotImplementedError(
				'Cannot process data with ndim > 3 (ndim = {})'.format(ndim))
		self.im_size = [im_size, ]
		if isinstance(self.pool_size, (list, tuple)):
			kernel_size_up = self.pool_size
		else:
			kernel_size_up = (self.pool_size,) * ndim
		if len(kernel_size_up) == 1:
			kernel_size_up = kernel_size_up[0]

		f_conv = getattr(nn, 'Conv{}D'.format(ndim))
		f_maxpool = getattr(nn, 'MaxPooling{}D'.format(ndim))
		f_upsamp = getattr(nn, 'UpSampling{}D'.format(ndim))

		cond_emb_layer = [[nn.Dense(units=tf.reduce_prod(im_size)), GELU(),]]
		if self.cond_params['network_name'].lower() in 'encoder':
			cond_emb_layer.append([Encoder_v3_noskip(**self.cond_params['network_kwargs']),])
		elif self.cond_params['network_name'].lower() in 'mlp':
			cond_emb_layer.append([nn.Dense(units=tf.reduce_prod(im_size)), GELU(),])
		elif self.cond_params['network_name'].lower() not in ('encoder', 'mlp'):
			raise NotImplementedError('Network name for condition embedding not recognized. '
			                          'Expected ``encoder`` or ``mlp`` (given {})'.
			                          format(self.cond_params['network_name']))

		# conv_out = []
		for di in range(self.depth):
			# Condition embedding (time and label)
			tmp_cond_layer = []
			for cond_i in range(2):
				if self.sinusoidal_cond_mlp and cond_i == 0:
					sin_emb = SinusoidalPosEmb(self.sin_emb_dim)
				else:
					sin_emb = Identity()
				tmp_cond_layer.append(Sequential([
					sin_emb, nn.Flatten() if self.cond_params['flag_flatten_input']
					else Identity()] + cond_emb_layer[cond_i] + \
					# nn.LayerNormalization(),
					[nn.Dense(units=tf.reduce_prod(im_size)),
					nn.Reshape(im_size + (-1,))
					 ], name="cond_{}_embedding".format(cond_i)))
			self.cond_mlp_down.append(tmp_cond_layer)

			num_units = self.num_filt_start * 2 ** di
			tmp_layer = []
			# Inner block
			if self.block_type == 'conv':
				tmp_layer.append(ConvBlock(ndim, num_units,
				                              **hlp.dict_merge({'conv_args': self.conv_args},
				                                           self.block_params)))
			else:
				raise NotImplementedError(
					'Block type unknown ({})'.format(self.block_type))
			# Downsampling
			if di < self.depth - 1:
				tmp_layer.append(f_maxpool(pool_size=kernel_size_up, padding='same'))
				im_size = tuple([size // self.pool_size for size in im_size])
				self.im_size.append(im_size)
				if self.dropout is not None:
					tmp_layer.append(nn.Dropout(self.dropout))
				else:
					tmp_layer.append(Identity())
			else:
				tmp_layer.append(Identity())  # fake maxpool
				tmp_layer.append(Identity())  # fake dropout

			self.downs.append(tmp_layer)

		for di in range(self.depth - 1, 0, -1):
			# Condition embedding
			tmp_cond_layer = []
			for cond_i in range(2):
				if self.sinusoidal_cond_mlp and cond_i == 0:
					sin_emb = SinusoidalPosEmb(self.sin_emb_dim)
				else:
					sin_emb = Identity()
				tmp_cond_layer.append(Sequential([
					sin_emb, nn.Flatten() if self.cond_params['flag_flatten_input']
					else Identity()] + cond_emb_layer[cond_i] + \
					# nn.LayerNormalization(),
					[nn.Dense(units=tf.reduce_prod(im_size)),
					nn.Reshape(im_size + (-1,))
					 ], name="cond_{}_embedding".format(cond_i)))
			self.cond_mlp_up.append(tmp_cond_layer)

			tmp_layer = []
			num_units = self.num_filt_start * 2 ** (di - 1)
			# Upsampling
			tmp_layer.append(f_upsamp(kernel_size_up))
			im_size = tuple([size * self.pool_size for size in im_size])
			self.im_size.append(im_size)
			tmp_layer.append(f_conv(num_units, kernel_size=kernel_size_up,
			                        **self.conv_args))
			if self.dropout is not None:
				tmp_layer.append(nn.Dropout(self.dropout))
			else:
				tmp_layer.append(Identity())
			# Inner block
			if self.block_type == 'conv':
				tmp_layer.append(ConvBlock(ndim, num_units,
				                              **hlp.dict_merge({'conv_args': self.conv_args},
				                                           self.block_params)))
			else:
				raise NotImplementedError(
					'Block type unknown ({})'.format(self.block_type))
			self.ups.append(tmp_layer)

		self.final_conv = f_conv(self.num_channels_out, kernel_size=(1,) * ndim,
		                         activation=self.final_activation, **self.conv_args)

	def call(self, inputs, time=None, condition=None, training=False, **kwargs):
		x = inputs

		if self.normalize_layer['feature'] is not None:
			x = self.normalize_layer['feature'](x)
		if self.normalize_layer['label'] is not None:
			condition = self.normalize_layer['label'](condition)

		conv_out = []
		for di, (conv, pool, dropout) in enumerate(self.downs):
			# Apply MLP and Concatenate condition
			time_cond = self.cond_mlp_down[di][0](time)
			label_cond = self.cond_mlp_down[di][1](condition)
			x = tf.concat([label_cond, time_cond, x], axis=-1,
			              name='concat_up_{}'.format(di))
			# Convolution
			x = conv(x, training=training)
			# Keep output for skip connection
			conv_out.append(x)
			# Downsample (or identity if last layer)
			x = pool(x, training=training)
			# Dropout (or identity if self.dropout is None)
			x = dropout(x, training=training)

		for di, (up, conv1, dropout, conv2) in enumerate(self.ups):
			# Apply MLP and Concatenate condition
			time_cond = self.cond_mlp_up[di][0](time)
			label_cond = self.cond_mlp_up[di][1](condition)
			x = tf.concat([label_cond, time_cond, x], axis=-1,
			              name='concat_up_{}'.format(di))
			# Upsample
			x = up(x, training=training)
			x = conv1(x, training=training)
			# Pre-process incoming skip connection
			if self.skip_conn_op is None:
				x_skip_in = conv_out[self.depth - 2 - di]
			else:
				raise NotImplementedError(
					'Skip connection pre-processing not implemented ({})'.
					format(self.skip_conn_op))
			# Skip connection (concat or add)
			if self.skip_conn_type == 'concat':
				x = nn.concatenate([x_skip_in, x])
			elif self.skip_conn_type == 'add':
				x = nn.add([x_skip_in, x])
			else:
				raise NotImplementedError(
					'Skip connection type unknown ({})'.
					format(self.skip_conn_type))
			# Post-process output of skip connection
			if self.skip_conn_post_op is not None:
				raise NotImplementedError(
					'Skip connection post-processing not implemented ({})'.
					format(self.skip_conn_post_op))

			# Dropout
			x = dropout(x, training=training)
			x = conv2(x, training=training)

		x = self.final_conv(x, training=training)

		if self.denormalize_layer['feature'] is not None:
			x = self.denormalize_layer['feature'](x)

		return x

	def get_config(self):
		base_config = super().get_config()
		base_config.update({
			"num_channels_out":    self.num_channels_out,
			"num_filt_start":      self.num_filt_start,
			"pool_size":           self.pool_size,
			"depth":               self.depth,
			"block_type":          self.block_type,
			"block_params":        self.block_params,
			"skip_conn_type":      self.skip_conn_type,
			"skip_conn_op":        self.skip_conn_op,
			"skip_conn_post_op":   self.skip_conn_post_op,
			"dropout":             self.dropout,
			"final_activation":    self.final_activation,
			"sin_emb_dim":         self.sin_emb_dim,
			"conv_args":           self.conv_args,
			'cond_params':         self.cond_params,
			'normalize_feature_dict': self.normalize_feature_dict,
			'normalize_label_dict':   self.normalize_label_dict,

		})

		return base_config





@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class Autoencoder(Layer):
	"""Combines the encoder and decoder into an end-to-end model for training."""

	def __init__(
			self, enc_size, latent_dim, dec_size, activation='relu',
			flag_flatten=True, flag_skip=False,
			normalize_feature_dict=None, normalize_label_dict=None,
			encoder_kwargs=None, decoder_kwargs=None, **kwargs):
		super().__init__()

		self.enc_size = enc_size
		self.dec_size = dec_size
		self.latent_dim = latent_dim
		self.activation = activation
		self.flag_flatten = flag_flatten
		self.flag_skip = flag_skip
		self.normalize_feature_dict = normalize_feature_dict or {}
		self.normalize_label_dict = normalize_label_dict or {}
		self.encoder_kwargs = encoder_kwargs
		self.decoder_kwargs = decoder_kwargs
		self.kwargs = kwargs

		self.encoder = None
		self.decoder = None
		self.ndim = None
		self.output_dim = None
		self._input_shape = None
		self.reshape_dim = None
		self.t_concat_shape = None
		self.normalize_layer = {}
		self.denormalize_layer = {}

		if (enc_size != dec_size[::-1]) and flag_skip:
			raise ValueError('``enc_size`` and ``dec_size`` must have the same values (reversed) '
			                 'to use additive skip connections. \n'
			                 '(Inputs: enc_size = {} , dec_size = {})'.format(enc_size, dec_size))

		# Create Normalization layers if normalize_*_dict is not None
		for name_i, normalize_dict in zip(['feature', 'label'],
		                                  [self.normalize_feature_dict,
		                                   self.normalize_label_dict]):
			lower_keys = list(map(str.lower, normalize_dict.keys()))
			self.normalize_layer[name_i] = None
			self.denormalize_layer[name_i] = None
			if ('mean' in lower_keys) and ('var' in lower_keys or 'std' in lower_keys):
				normalize_kwargs = {
					'mean': np.array(normalize_dict.get('mean')),
					'var': np.array(normalize_dict.get('var', None)) if 'var' in lower_keys \
					else (np.array(normalize_dict.get('std', None)) ** 2)
				}
			elif ('min' in lower_keys) and ('max' in lower_keys):
				normalize_kwargs = {
					'min': np.array(normalize_dict.get('min')),
					'max': np.array(normalize_dict.get('max'))
				}
			else:
				raise ValueError('Normalization method for {} not recognized. Normalization dict '
				                 'should have named variables ``min`` and ``max`` or ``mean`` and '
				                 '``var`` (or ``std``). Given {}'.format(name_i, normalize_dict.keys()))

			# Create a Normalization layer and set its internal state using passed mean and variance
			self.normalize_layer[name_i] = NormalizationLayer(**normalize_kwargs)
			self.denormalize_layer[name_i] = NormalizationLayer(invert=True, **normalize_kwargs)


	def build(self, input_shape=None):
		ndim = len(input_shape) - 2
		self.ndim = ndim
		self._input_shape = input_shape
		self.reshape_dim = (-1,) + (1,) * (self.ndim + 1)
		self.t_concat_shape = [1, ] + input_shape[1:-1] + [1, ]

		if self.flag_flatten:
			self.output_dim = np.prod(input_shape[1:])
		else:
			self.output_dim = input_shape[-1]

		self.encoder = Encoder_v3(enc_size=self.enc_size, latent_dim=self.latent_dim,
		                          activation=self.activation, flag_skip=self.flag_skip,
		                          **self.encoder_kwargs)
		self.decoder = Decoder_v3(dec_size=self.dec_size, output_dim=self.output_dim,
		                          activation=self.activation, flag_skip=self.flag_skip,
		                          **self.decoder_kwargs)

	def call(self, inputs, time=None, condition=None, training=False, **kwargs):

		x1 = inputs
		if self.normalize_layer['feature'] is not None:
			x1 = self.normalize_layer['feature'](x1)
		if self.normalize_layer['label'] is not None:
			condition = self.normalize_layer['label'](condition)

		t = tf.cast(tf.reshape(time, self.reshape_dim), x1.dtype)

		if self.flag_flatten:
			x1 = nn.Flatten()(x1)
			condition = nn.Flatten()(condition)
			t = nn.Flatten()(t)
		else:
			t = tf.tile(t, self.t_concat_shape)
		x = tf.concat([x1, condition, t], axis=-1, name='concat_input')
		z, h = self.encoder(x, training=training)
		reconstructed = self.decoder([
			tf.concat([z, condition], axis=-1, name='concat_latent'), h],
			training=training)

		if self.flag_flatten:
			reconstructed = nn.Reshape(self._input_shape[1:])(reconstructed)

		if self.denormalize_layer['feature'] is not None:
			reconstructed = self.denormalize_layer['feature'](reconstructed)

		return reconstructed

	def get_config(self):
		base_config = super().get_config()
		base_config.update({
			'enc_size':         self.enc_size,
			'dec_size':         self.dec_size,
			'latent_dim':       self.latent_dim,
			'activation':       self.activation,
			'flag_flatten':           self.flag_flatten,
			'flag_skip':              self.flag_skip,
			'normalize_feature_dict': self.normalize_feature_dict,
			'normalize_label_dict':   self.normalize_label_dict,
			'encoder_kwargs':   self.encoder_kwargs,
			'decoder_kwargs':   self.decoder_kwargs
		})
		return base_config
