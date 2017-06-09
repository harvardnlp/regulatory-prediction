import tensorflow as tf

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
    	with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.scalar('abs_sum', tf.reduce_sum(tf.abs(var)))
			tf.summary.histogram('histogram', var)

def _batch_norm(x, is_training, decay_rate, scope):
	bn = tf.contrib.layers.batch_norm(inputs=x, decay=decay_rate, scope=scope, center=True, scale=True, epsilon=1e-10, is_training=is_training, activation_fn=None, updates_collections=None)
	with tf.variable_scope(scope, reuse=True):
		variable_summaries(tf.get_variable("moving_mean"), "moving_mean")
		variable_summaries(tf.get_variable("moving_variance"), "moving_variance")
		variable_summaries(tf.get_variable("beta"), "beta")
		variable_summaries(tf.get_variable("gamma"), "gamma")
	return bn