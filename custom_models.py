import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Conv2DTranspose, concatenate, Layer
import tensorflow.keras.layers as nn
import inspect
import numpy as np

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)
TF_DTYPE = eval('tf.' + DTYPE)
NP_DTYPE = eval('np.' + DTYPE)

import custom_layers as cl
import helpers as hlp

# Clear previously registered custom objects from this module
del_key = []
for key in tf.keras.utils.get_custom_objects().keys():
	if 'CustomModels' in key:
		del_key.append(key)
for del_key_i in del_key:
	tf.keras.utils.get_custom_objects().pop(del_key_i)


## Networks and Models

@tf.keras.utils.register_keras_serializable(package="CustomModels")
class UnetDiffConditional(Model):
	"""Diffusion model directly built with U-Net architecture."""

	def __init__(self, num_channels_out=1, num_filt_start=64,
	             pool_size=2, depth=5, block_type='conv', block_params=None,
	             skip_conn_type='concat', skip_conn_op=None,
	             skip_conn_post_op=None, dropout=None,
	             final_activation=None, sinusoidal_cond_mlp=True,
	             sin_emb_dim=64, timesteps=200, noise_schedule=None, **kwargs):
		super(UnetDiffConditional, self).__init__()

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
		self.sinusoidal_cond_mlp = sinusoidal_cond_mlp
		# self.n_condition = n_condition  # number of conditions to embed (e.g., time and digit)

		self.timesteps = timesteps
		self.beta_start = 1e-4 if noise_schedule is None else noise_schedule['beta_start']
		self.beta_end = 2e-2 if noise_schedule is None else noise_schedule['beta_end']

		self.beta = np.linspace(self.beta_start, self.beta_end, self.timesteps,
		                        dtype=NP_DTYPE)
		self.alpha = 1 - self.beta
		self.alpha_bar = np.cumprod(self.alpha, 0, dtype=NP_DTYPE)
		self.alpha_bar = np.concatenate((np.array([1.], dtype=NP_DTYPE),
		                                 self.alpha_bar[:-1]), axis=0)
		self.sqrt_alpha_bar = np.sqrt(self.alpha_bar, dtype=NP_DTYPE)
		self.sqrt_one_minus_alpha_bar = np.sqrt(1 - self.alpha_bar, dtype=NP_DTYPE)

		self.conv_args = {'padding': 'same'}
		self.ndim = None
		self.im_size = None

		# Create layers
		self.downs = []
		self.ups = []

		# Conditional embedding
		self.cond_mlp_down = []
		self.cond_mlp_up = []

	def build(self, input_shape):

		ndim = len(input_shape) - 2
		self.ndim = ndim
		if ndim == 1:
			im_size = (input_shape[1],)
			self.reshape_dim = (-1, 1, 1)
		elif ndim == 2:
			im_size = (input_shape[1], input_shape[2])
			self.reshape_dim = (-1, 1, 1, 1)
		elif ndim == 3:
			im_size = (input_shape[1], input_shape[2], input_shape[3])
			self.reshape_dim = (-1, 1, 1, 1, 1)
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

		# conv_out = []
		for di in range(self.depth):
			# Condition embedding (time and label)
			tmp_cond_layer = []
			for _ in range(2):
				if self.sinusoidal_cond_mlp:
					tmp_cond_layer.append(Sequential([
						cl.SinusoidalPosEmb(self.sin_emb_dim),
						nn.Dense(units=tf.reduce_prod(im_size)),
						cl.GELU(),
						nn.Dense(units=tf.reduce_prod(im_size)),
						nn.Reshape(im_size + (1,))
					], name="cond_embeddings"))
				else:
					tmp_cond_layer.append(Sequential([
						nn.Flatten(),
						nn.Dense(units=tf.reduce_prod(im_size)),
						cl.GELU(),
						# nn.LayerNormalization(),
						nn.Dense(units=tf.reduce_prod(im_size)),
						nn.Reshape(im_size + (1,))
					]))
			self.cond_mlp_down.append(tmp_cond_layer)

			num_units = self.num_filt_start * 2 ** di
			tmp_layer = []
			# Inner block
			if self.block_type == 'conv':
				tmp_layer.append(cl.ConvBlock(ndim, num_units,
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
					tmp_layer.append(cl.Identity())
			else:
				tmp_layer.append(cl.Identity())  # fake maxpool
				tmp_layer.append(cl.Identity())  # fake dropout

			self.downs.append(tmp_layer)

		for di in range(self.depth - 1, 0, -1):
			# Condition embedding
			tmp_cond_layer = []
			for _ in range(2):
				if self.sinusoidal_cond_mlp:
					tmp_cond_layer.append(Sequential([
						cl.SinusoidalPosEmb(self.sin_emb_dim),
						nn.Dense(units=tf.reduce_prod(im_size)),
						cl.GELU(),
						nn.Dense(units=tf.reduce_prod(im_size)),
						nn.Reshape(im_size + (1,))
					], name="cond_embeddings"))
				else:
					tmp_cond_layer.append(Sequential([
						nn.Flatten(),
						nn.Dense(units=tf.reduce_prod(im_size)),
						cl.GELU(),
						# nn.LayerNormalization(),
						nn.Dense(units=tf.reduce_prod(im_size)),
						nn.Reshape(im_size + (1,))
					]))
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
				tmp_layer.append(cl.Identity())
			# Inner block
			if self.block_type == 'conv':
				tmp_layer.append(cl.ConvBlock(ndim, num_units,
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

		return x

	def make_summary(self, input_shape, **kwargs):
		""" Replace default summary() method. (The default summary() method
		has a glitch and sometimes does not show output shapes.)"""
		x = tf.keras.Input(shape=input_shape)
		t = tf.keras.Input(shape=())
		label = tf.keras.Input(shape=())
		return tf.keras.Model(inputs=[x],
		                      outputs=self.call(x, time=t, condition=label)).summary(
			expand_nested=True, **kwargs)

	def compile(self, **kwargs):
		super().compile(**kwargs)
		self.noise_loss_tracker = tf.keras.metrics.Mean(name="noise_loss")

	@property
	def metrics(self):
		return [self.noise_loss_tracker, ]

	def train_step(self, data):
		images = data[0]
		# if len(data) < 2:
		# 	condition = None
		# else:
		condition = data[1]
		# print(tf.shape(images))
		# print(tf.shape(condition))
		images_shape = tf.shape(images)
		batch_size = tf.shape(images)[0]

		t = tf.random.uniform(shape=[batch_size],
		                      minval=0, maxval=self.timesteps,
		                      dtype=tf.int64)

		# noised_image, noise = forward_noise_tf(rng, images, timestep_values)
		noise = tf.random.normal(shape=images_shape, dtype=TF_DTYPE)
		sqrt_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_alpha_bar, t, axis=0), self.reshape_dim)
		sqrt_one_minus_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_one_minus_alpha_bar, t, axis=0), self.reshape_dim)
		noised_image = sqrt_alpha_bar_t * images + sqrt_one_minus_alpha_bar_t * noise
		with tf.GradientTape() as tape:
			prediction = self.call(noised_image, time=t, condition=condition, training=True)
			noise_loss = tf.math.reduce_mean((noise - prediction) ** 2)

		gradients = tape.gradient(noise_loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		self.noise_loss_tracker.update_state(noise_loss)

		return {m.name: m.result() for m in self.metrics}

	def ddpm(self, x_t, time, condition=None):
		""" Predicts x^{t-1} based on x^{t} using the DDPM model.
		Use in a for loop from T to 1 to retrieve input from noise."""

		alpha_t = tf.reshape(tf.gather(self.alpha, time, axis=0), self.reshape_dim)
		sqrt_one_minus_alpha_bar_t = tf.reshape(tf.gather(
			self.sqrt_one_minus_alpha_bar, time, axis=0), self.reshape_dim)
		pred_noise = self.call(x_t, time=time, condition=condition, training=False)

		eps_coef = (1 - alpha_t) / sqrt_one_minus_alpha_bar_t
		mean = (1 / tf.sqrt(alpha_t)) * (x_t - eps_coef * pred_noise)

		var = tf.reshape(tf.gather(self.beta, time, axis=0), self.reshape_dim)
		z = tf.random.normal(shape=tf.shape(x_t), dtype=x_t.dtype)

		return mean + tf.sqrt(var) * z


## Diffusion Model

@tf.keras.utils.register_keras_serializable(package="CustomModels")
class DiffusionModel(Model):
	"""
	Generic diffusion model class with built-in training method.
	Can be used with any type of input network, so long they
	take as argument the input Tensor ``x``, time embedding
	``time`` and optional condition (for conditional generative
	networks) ``condition``.
	"""

	def __init__(self, timesteps=200, noise_schedule=None,
	             network=None, ndim=None, **kwargs):
		"""
		Parameters
		----------

		:param timesteps: (int) Number of timestep for the diffusion process.
		:param noise_schedule: (dict) Dictionary containing
		``{'beta_start', 'beta_end'}`` corresponding to the low
		and high value of beta. The scheduled noise alpha is then
		calculated from beta as ``alpha = 1 - beta``. Defaults to
		``{'beta_start': 1e-4, 'beta_end': 2e-2}``
		:param network: (tf.keras.layers.Layer or Model) Tensorflow Neural network
		to process the data (e.g., U-Net). The model has to take **at least
		1 named input** in addition to the input data to process:
		``time`` (int), the current timestep of the diffusion process to be
		embedded in the network. A condition can also be optionally passed
		for conditional generative networks, using the keyword ``condition``.
		Note: the Layer must be already built and ideally has an ``ndim``
		attribute (``ndim = len(input_shape) - 2``). If the network uses
		layers such as Dropout or Batch Norm, make sure to pass the
		``training`` argument to these layers.
		:param ndim: (int) Rank of the input data, excluding batch and channel
		dimensions: ``len(input_shape) - 2`` (e.g, (batch_size, M, N,
		channel) --> ndim = 2). Fetched from network.ndim or network.input_shape
		if attribute exists.

		kwargs
		----------

		``ema_decay``: decay for the exponential moving average when updating
		trainable weights
		"""
		super(DiffusionModel, self).__init__()

		self.timesteps = timesteps
		self.noise_schedule = noise_schedule
		self.beta_start = 1e-4 if noise_schedule is None else noise_schedule.get('beta_start', 1e-4)
		self.beta_end = 2e-2 if noise_schedule is None else noise_schedule.get('beta_end', 2e-2)
		self.offset_s = 0.008 if noise_schedule is None else noise_schedule.get('offset_s', 0.008)
		self.max_beta = 0.999 if noise_schedule is None else noise_schedule.get('max_beta', 0.999)

		self.schedule_name = 'linear' if noise_schedule is None else noise_schedule.get('schedule_name', 'linear')

		self.beta = hlp.get_beta_schedule(self.schedule_name, self.timesteps,
		                                  beta_start=self.beta_start,
		                                  beta_end=self.beta_end,
		                                  offset_s=self.offset_s,
		                                  max_beta=self.max_beta)

		self.alpha = 1 - self.beta
		self.alpha_bar = np.cumprod(self.alpha, 0, dtype=NP_DTYPE)
		self.alpha_bar = np.concatenate((np.array([1.], dtype=NP_DTYPE),
		                                 self.alpha_bar[:-1]), axis=0)
		self.sqrt_alpha_bar = np.sqrt(self.alpha_bar, dtype=NP_DTYPE)
		self.sqrt_one_minus_alpha_bar = np.sqrt(1 - self.alpha_bar, dtype=NP_DTYPE)

		ema_decay = kwargs.get('ema_decay', 0.9999)
		self.ema = tf.train.ExponentialMovingAverage(decay=ema_decay)

		# Check input network
		if network is not None:
			self.network = network
		else:
			raise ValueError('Input network is required to create model.')

		# Check operating number of dimension
		if ndim is not None:
			self.ndim = ndim
		else:
			if hasattr(network, 'ndim'):
				self.ndim = network.ndim
			elif hasattr(network, 'input_shape'):
				self.ndim = len(input_shape) - 2
			else:
				raise AttributeError('``ndim`` was not passed as argument. '
				                     'Could not be retrieved from input network'
				                     ' {} (``input_shape`` attribute '
				                     'does not exist).'.format(network.name))

		self.reshape_dim = (-1,) + (1,) * (self.ndim + 1)

		if 'condition' in inspect.getfullargspec(network.call)[0]:
			self.flag_condition = True
		else:
			self.flag_condition = False

	def call(self, inputs, time=None, condition=None, **kwargs):
		"""Model call"""
		cond_kwargs = {}
		if self.flag_condition:
			cond_kwargs = {'condition': condition}
		output = self.network(inputs, time=time, **cond_kwargs, **kwargs)
		return output

	def make_summary(self, input_shape, cond_shape=None, **kwargs):
		""" Replace default summary() method. Shapes do not include batch size.
		(The default summary() method
		has a glitch and sometimes does not show output shapes.)
		May not work with all provided networks (and return Graph disconnected error)."""
		x = tf.keras.Input(shape=input_shape)
		t = tf.keras.Input(shape=())
		cond_kwargs = {}
		if cond_shape is not None:
			label = tf.keras.Input(shape=cond_shape)
			cond_kwargs = {'condition': label}
		return tf.keras.Model(inputs=[x],
		                      outputs=self.call(x, time=t, **cond_kwargs)).summary(
			expand_nested=True, **kwargs)

	def compile(self, **kwargs):
		super().compile(**kwargs)
		self.loss_tracker = tf.keras.metrics.Mean(name="loss")
		self.noise_loss_tracker = tf.keras.metrics.Mean(name="noise_loss")

	@property
	def metrics(self):
		return [self.loss_tracker, self.noise_loss_tracker]

	def compute_loss(self, x=None, y=None, y_pred=None):
		"""Compute loss (e.g., l2 norm between noise and predicted noise).
		Regularization losses (as well as any other loss in self.network
		passed in init) are also calculated here."""
		noise_loss = self.loss(y_true=y, y_pred=y_pred)
		reg_loss = 0.
		if len(self.losses) > 0:
			reg_loss = tf.add_n(self.losses)
		loss = noise_loss + reg_loss
		return loss, noise_loss

	# @tf.function
	def train_step(self, data):
		""" Override training step in model.fit() for the diffusion process.
		A loss needs to be provided when compiling the model (e.g., l2 norm)."""
		images = data[0]
		if len(data) < 2:
			condition = None
		else:
			condition = data[1]
		images_shape = tf.shape(images)
		batch_size = tf.shape(images)[0]

		t = tf.random.uniform(shape=[batch_size],
		                      minval=0, maxval=self.timesteps,
		                      dtype=tf.int64)
		noise = tf.random.normal(shape=images_shape, dtype=TF_DTYPE)

		# Retrieve the current timestep for each batch, and reshape for broadcast
		sqrt_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_alpha_bar, t, axis=0), self.reshape_dim)
		sqrt_one_minus_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_one_minus_alpha_bar, t, axis=0), self.reshape_dim)
		noised_image = sqrt_alpha_bar_t * images + sqrt_one_minus_alpha_bar_t * noise

		# Check if condition is an input argument of the network
		cond_kwargs = {}
		if self.flag_condition:
			cond_kwargs = {'condition': condition}

		with tf.GradientTape() as tape:
			prediction = self.call(noised_image, time=t, training=True, **cond_kwargs)
			loss, noise_loss = self.compute_loss(y=noise, y_pred=prediction)
			# scaled_loss = self.optimizer.get_scaled_loss(loss)
			# noise_loss = tf.math.reduce_mean((noise - prediction) ** 2)  # Default loss for diffusion models

		# scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
		# gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
		# Apply exponential moving average for stability
		#  (https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)
		self.ema.apply(self.trainable_variables)

		self.loss_tracker.update_state(loss)
		self.noise_loss_tracker.update_state(noise_loss)

		return {m.name: m.result() for m in self.metrics}

	# @tf.function
	def test_step(self, data):
		""" Override default validation step in model.evaluate() for the diffusion process.
		A loss needs to be provided when compiling the model (e.g., l2 norm)."""
		images = data[0]
		if len(data) < 2:
			condition = None
		else:
			condition = data[1]
		images_shape = tf.shape(images)
		batch_size = tf.shape(images)[0]

		t = tf.random.uniform(shape=[batch_size],
		                      minval=0, maxval=self.timesteps,
		                      dtype=tf.int64)
		noise = tf.random.normal(shape=images_shape, dtype=TF_DTYPE)

		# Retrieve the current timestep for each batch, and reshape for broadcast
		sqrt_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_alpha_bar, t, axis=0), self.reshape_dim)
		sqrt_one_minus_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_one_minus_alpha_bar, t, axis=0), self.reshape_dim)
		noised_image = sqrt_alpha_bar_t * images + sqrt_one_minus_alpha_bar_t * noise

		# Check if condition is an input argument of the network
		cond_kwargs = {}
		if self.flag_condition:
			cond_kwargs = {'condition': condition}

		prediction = self.call(noised_image, time=t, training=False, **cond_kwargs)
		loss, noise_loss = self.compute_loss(y=noise, y_pred=prediction)

		self.loss_tracker.update_state(loss)
		self.noise_loss_tracker.update_state(noise_loss)

		return {m.name: m.result() for m in self.metrics}

	@tf.function
	def ddpm(self, x_t, time, condition=None, variance='beta'):
		""" Predicts x^{t-1} based on x^{t} using the DDPM model.
		Use in a for loop from T to 1 to retrieve input from noise."""

		alpha_t = tf.reshape(tf.gather(self.alpha, time, axis=0), self.reshape_dim)
		alpha_bar_t_prev = tf.reshape(tf.gather(self.alpha_bar, time - 1, axis=0), self.reshape_dim)
		alpha_bar_t = tf.reshape(tf.gather(self.alpha_bar, time, axis=0), self.reshape_dim)
		sqrt_one_minus_alpha_bar_t = tf.reshape(tf.gather(
			self.sqrt_one_minus_alpha_bar, time, axis=0), self.reshape_dim)
		pred_noise = self.call(x_t, time=time, condition=condition, training=False)

		eps_coef = (1 - alpha_t) / sqrt_one_minus_alpha_bar_t
		mean = (1 / tf.sqrt(alpha_t)) * (x_t - eps_coef * pred_noise)

		beta_t = tf.reshape(tf.gather(self.beta, time, axis=0), self.reshape_dim)
		if variance in 'beta':
			var = beta_t
		elif variance in 'beta_tilde':
			var = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t
		else:
			raise NotImplementedError()
		z = tf.random.normal(shape=tf.shape(x_t), dtype=x_t.dtype)

		return mean + tf.sqrt(var) * z

	@tf.function
	def ddim(self, x_t, time, condition=None, eta=0.):
		"""Extension of DDPM that uses non-markovian process for inference.
		Uses a rate :math:`0 < \\eta \\leq 1` to control stochasticity. Produce
		sub-sequences :math:`{\\tau}_i \\in [0, ..., T]` and generate samples with \"less\"
		stochasticity (:math:`\\eta < 1`) or fully deterministic (:math:`\\eta = 0`).
		Note that in the case where :math:`\\eta = 1`, we get back to the DDPM model.
		"""

		alpha_t_bar = tf.reshape(tf.gather(self.alpha_bar, time, axis=0), self.reshape_dim)
		alpha_t_bar_minus_one = tf.reshape(tf.gather(self.alpha_bar, time - 1, axis=0), self.reshape_dim)
		sigma_t = eta * tf.sqrt((1 - alpha_t_bar_minus_one) / (1 - alpha_t_bar)) * \
		          tf.sqrt(1 - alpha_t_bar / alpha_t_bar_minus_one)

		pred_noise = self.call(x_t, time=time, condition=condition, training=False)

		pred_x0 = (x_t - tf.sqrt(1 - alpha_t_bar) * pred_noise) / tf.sqrt(alpha_t_bar)
		pred_x0 = tf.sqrt(alpha_t_bar_minus_one) * pred_x0

		x_t_direction = tf.sqrt(1 - alpha_t_bar_minus_one - tf.square(sigma_t)) * pred_noise
		eps_t = tf.random.normal(shape=tf.shape(x_t), dtype=x_t.dtype)

		return pred_x0 + x_t_direction + (sigma_t * eps_t)

	def get_config(self):
		"""Serialize object"""
		config = super().get_config()
		config.update({
			'timesteps':      self.timesteps,
			'noise_schedule': self.noise_schedule,
			'ndim':           self.ndim,
			'network':        tf.keras.layers.serialize(self.network)
		})
		return config

	@classmethod
	def from_config(cls, config):
		"""Create object from configuration"""
		config["network"] = tf.keras.layers.deserialize(config["network"])
		return cls(**config)
