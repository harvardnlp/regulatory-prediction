import numpy as np
import tensorflow as tf
import sys
import gc
import time
#from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tf_utils import variable_summaries, _batch_norm
from custom_ops import atrous_pool2d

def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  w_var = tf.Variable(initial, name=name)
  if name != None:
  	variable_summaries(w_var, name)
  return w_var

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  b_var = tf.Variable(initial, name=name)
  if name != None:
  	variable_summaries(b_var, name)
  return b_var

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  
def atrous_conv2d(x, W, rate):
	return tf.nn.atrous_conv2d(x, W, rate, padding='SAME')

def max_pool(x, kH):
  return tf.nn.max_pool(x, ksize=[1, kH, 1, 1],
                        strides=[1, 1, 1, 1], padding='SAME')

def atrous_pool(x, kH, dilation_rate):
	return atrous_pool2d(x, ksize=[1, kH, 1, 1], rate=dilation_rate, padding="SAME", pooling_type="MAX")
# return tf.nn.pool(x, dilation_rate=[dilation_rate, 1], window_shape=[kH, 1],
#                       padding='VALID', pooling_type="MAX")

def dilated_convolution_model(x, y_, dropout_keep_prob, batch_size, noutputs, dim_reduction=10, nkernels=[128, 240, 50], hidden=125):

	#dim_reduction = 10
	#nkernels = [128, 240, 50]
	#hidden = 125
	embedding_size = 4
	W = {
		"conv1": weight_variable([1*dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		"conv2": weight_variable([dim_reduction, 1, nkernels[0], nkernels[1]], "W_conv2"),
		"conv3": weight_variable([dim_reduction, 1, nkernels[1], nkernels[2]]),
		"conv4": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv5": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv6": weight_variable([1*dim_reduction, 1, nkernels[2], nkernels[2]], "W_conv6"),
		#"conv7": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"hid1": weight_variable([nkernels[2], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"conv2": bias_variable([nkernels[1]]),
		"conv3": bias_variable([nkernels[2]]),
		"conv4": bias_variable([nkernels[2]]),
		"conv5": bias_variable([nkernels[2]]),
		"conv6": bias_variable([nkernels[2]]),
		#"conv7": bias_variable([nkernels[2]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)

	# add the convolutional layers
	h_conv1 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(embedded_image, W["conv1"], strides=[1, 1*dim_reduction, 1, 1], padding='SAME') + b["conv1"]), dropout_keep_prob)
	h_conv2 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv1, W["conv2"], 3) + b["conv2"]), dropout_keep_prob)
	h_conv3 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv2, W["conv3"], 9) + b["conv3"]), dropout_keep_prob)
	h_conv4 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv3, W["conv4"], 27) + b["conv4"]), dropout_keep_prob)
	h_conv5 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv4, W["conv5"], 81) + b["conv5"]), dropout_keep_prob)
	h_conv6 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d_transpose(h_conv5, W["conv6"], [batch_size, 25000, 1, nkernels[2]], [1, 1*dim_reduction, 1, 1]) + b["conv6"]), dropout_keep_prob)
	flattened = tf.reshape(tf.squeeze(h_conv6), [batch_size*25000, nkernels[2]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]
	
	return outputs, W, b, embed, [h_conv6]

def dilated_convolution_model_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, dim_reduction=10, nkernels=[128, 240, 50], hidden=125):

	# dim_reduction = 10
	# nkernels = [128, 240, 50]
	# hidden = 125
	embedding_size = 4
	W = {
		"conv1": weight_variable([1*dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		"conv2": weight_variable([dim_reduction, 1, nkernels[0], nkernels[1]]),
		"conv3": weight_variable([dim_reduction, 1, nkernels[1], nkernels[2]], "W_conv3"),
		"conv4": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv5": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv6": weight_variable([1**dim_reduction, 1, nkernels[2], nkernels[2]], "W_conv6"),
		#"conv7": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"hid1": weight_variable([nkernels[2], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"conv2": bias_variable([nkernels[1]]),
		"conv3": bias_variable([nkernels[2]]),
		"conv4": bias_variable([nkernels[2]]),
		"conv5": bias_variable([nkernels[2]]),
		"conv6": bias_variable([nkernels[2]]),
		#"conv7": bias_variable([nkernels[2]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)

	# add the convolutional layers
	h_conv1 = tf.nn.conv2d(embedded_image, W["conv1"], strides=[1, 1*dim_reduction, 1, 1], padding='SAME') + b["conv1"]
	h_conv1_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv1, is_training, decay_rate, "conv1-norm")), dropout_keep_prob)
	h_conv2 = atrous_conv2d(h_conv1_norm, W["conv2"], 3) + b["conv2"]
	h_conv2_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv2, is_training, decay_rate, "conv2-norm")), dropout_keep_prob)
	h_conv3 = atrous_conv2d(h_conv2_norm, W["conv3"], 9) + b["conv3"]
	h_conv3_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv3, is_training, decay_rate, "conv3-norm")), dropout_keep_prob)
	h_conv4 = atrous_conv2d(h_conv3_norm, W["conv4"], 27) + b["conv4"]
	h_conv4_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv4, is_training, decay_rate, "conv4-norm")), dropout_keep_prob)
	h_conv5 = atrous_conv2d(h_conv4_norm, W["conv5"], 81) + b["conv5"]
	h_conv5_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv5, is_training, decay_rate, "conv5-norm")), dropout_keep_prob)

	h_conv6 = tf.nn.conv2d_transpose(h_conv5_norm, W["conv6"], [batch_size, 25000, 1, nkernels[2]], [1, 1*dim_reduction, 1, 1]) + b["conv6"]
	h_conv6_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv6, is_training, decay_rate, "conv6-norm")), dropout_keep_prob)
	flattened = tf.reshape(tf.squeeze(h_conv6_norm), [batch_size*25000, nkernels[2]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]

	return outputs, W, b, embed, [h_conv6_norm]


def dilated_convolution_with_pooling(x, y_, dropout_keep_prob, batch_size, noutputs, dim_reduction=10, nkernels=[128, 240, 50], hidden=125):

	# dim_reduction = 10
	# nkernels = [128, 240, 50]
	# hidden = 125
	embedding_size = 4
	filter_height = 5
	W = {
		"conv1": weight_variable([dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		"conv2": weight_variable([dim_reduction, 1, nkernels[0], nkernels[1]]),
		"conv3": weight_variable([dim_reduction, 1, nkernels[1], nkernels[2]], "W_conv3"),
		"conv4": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv5": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv6": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]], "W_conv6"),
		# "conv7": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"hid1": weight_variable([nkernels[2], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"conv2": bias_variable([nkernels[1]]),
		"conv3": bias_variable([nkernels[2]]),
		"conv4": bias_variable([nkernels[2]]),
		"conv5": bias_variable([nkernels[2]]),
		"conv6": bias_variable([nkernels[2]]),
		"conv7": bias_variable([nkernels[2]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)

	# add the convolutional layers
	h_conv1 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(embedded_image, W["conv1"], strides=[1, dim_reduction, 1, 1], padding='SAME') + b["conv1"]), dropout_keep_prob)
	h_conv1_pooled = max_pool(h_conv1, filter_height)
	h_conv2 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv1_pooled, W["conv2"], 3) + b["conv2"]), dropout_keep_prob)
	h_conv2_pooled = max_pool(h_conv2, filter_height)
	h_conv3 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv2_pooled, W["conv3"], 9) + b["conv3"]), dropout_keep_prob)
	h_conv4 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv3, W["conv4"], 27) + b["conv4"]), dropout_keep_prob)
	h_conv4_pooled = max_pool(h_conv4, filter_height)
	h_conv5 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv4_pooled, W["conv5"], 81) + b["conv5"]), dropout_keep_prob)
	h_conv6 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d_transpose(h_conv5, W["conv6"], [batch_size, 25000, 1, nkernels[2]], [1, dim_reduction, 1, 1]) + b["conv6"]), dropout_keep_prob)
	flattened = tf.reshape(tf.squeeze(h_conv6), [batch_size*25000, nkernels[2]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]
	return outputs, W, b, embed, [h_conv6]

def dilated_convolution_with_pooling_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, dim_reduction=10, nkernels=[128, 240, 50], hidden=125):

	# dim_reduction = 10
	# nkernels = [128, 240, 50]
	# hidden = 125
	embedding_size = 4
	filter_height = 5
	W = {
		"conv1": weight_variable([dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		"conv2": weight_variable([dim_reduction, 1, nkernels[0], nkernels[1]]),
		"conv3": weight_variable([dim_reduction, 1, nkernels[1], nkernels[2]], "W_conv3"),
		"conv4": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv5": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv6": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]], "W_conv6"),
		# "conv7": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"hid1": weight_variable([nkernels[2], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"conv2": bias_variable([nkernels[1]]),
		"conv3": bias_variable([nkernels[2]]),
		"conv4": bias_variable([nkernels[2]]),
		"conv5": bias_variable([nkernels[2]]),
		"conv6": bias_variable([nkernels[2]]),
		"conv7": bias_variable([nkernels[2]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)

	# add the convolutional layers

	h_conv1 = tf.nn.conv2d(embedded_image, W["conv1"], strides=[1, dim_reduction, 1, 1], padding='SAME') + b["conv1"]
	h_conv1_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv1, is_training, decay_rate, "conv1-norm")), dropout_keep_prob)
	h_conv1_pooled = max_pool(h_conv1_norm, filter_height)

	h_conv2 = atrous_conv2d(h_conv1_pooled, W["conv2"], 3) + b["conv2"]
	h_conv2_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv2, is_training, decay_rate, "conv2-norm")), dropout_keep_prob)
	h_conv2_pooled = max_pool(h_conv2_norm, filter_height)

	h_conv3 = atrous_conv2d(h_conv2_pooled, W["conv3"], 9) + b["conv3"]
	h_conv3_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv3, is_training, decay_rate, "conv3-norm")), dropout_keep_prob)

	h_conv4 = atrous_conv2d(h_conv3_norm, W["conv4"], 27) + b["conv4"]
	h_conv4_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv4, is_training, decay_rate, "conv4-norm")), dropout_keep_prob)
	h_conv4_pooled = max_pool(h_conv4_norm, filter_height)

	h_conv5 = atrous_conv2d(h_conv4_pooled, W["conv5"], 81) + b["conv5"]
	h_conv5_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv5, is_training, decay_rate, "conv5-norm")), dropout_keep_prob)

	h_conv6 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d_transpose(h_conv5_norm, W["conv6"], [batch_size, 25000, 1, nkernels[2]], [1, dim_reduction, 1, 1]) + b["conv6"]), dropout_keep_prob)
	h_conv6_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv6, is_training, decay_rate, "conv6-norm")), dropout_keep_prob)
	
	flattened = tf.reshape(tf.squeeze(h_conv6_norm), [batch_size*25000, nkernels[2]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]
	return outputs, W, b, embed, [h_conv6_norm]


def dilated_convolution_with_dilated_pooling(x, y_, dropout_keep_prob, batch_size, noutputs, dim_reduction=10, nkernels=[128, 240, 50], hidden=125):

	# dim_reduction = 10
	# nkernels = [128, 240, 50]
	# hidden = 125
	embedding_size = 4
	filter_height = 5
	W = {
		"conv1": weight_variable([dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		"conv2": weight_variable([dim_reduction, 1, nkernels[0], nkernels[1]]),
		"conv3": weight_variable([dim_reduction, 1, nkernels[1], nkernels[2]], "W_conv3"),
		"conv4": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv5": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv6": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]], "W_conv6"),
		# "conv7": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"hid1": weight_variable([nkernels[2], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"conv2": bias_variable([nkernels[1]]),
		"conv3": bias_variable([nkernels[2]]),
		"conv4": bias_variable([nkernels[2]]),
		"conv5": bias_variable([nkernels[2]]),
		"conv6": bias_variable([nkernels[2]]),
		"conv7": bias_variable([nkernels[2]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)

	# add the convolutional layers
	h_conv1 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(embedded_image, W["conv1"], strides=[1, dim_reduction, 1, 1], padding='SAME') + b["conv1"]), dropout_keep_prob)
	h_conv2 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv1, W["conv2"], 3) + b["conv2"]), dropout_keep_prob)
	h_conv3 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv2, W["conv3"], 9) + b["conv3"]), dropout_keep_prob)
	h_conv4 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv3, W["conv4"], 27) + b["conv4"]), dropout_keep_prob)
	h_conv4_pooled = atrous_pool(h_conv4, filter_height, 27)
	h_conv5 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv4_pooled, W["conv5"], 81) + b["conv5"]), dropout_keep_prob)
	h_conv5_pooled = atrous_pool(h_conv5, filter_height, 81)
	h_conv6 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d_transpose(h_conv5_pooled, W["conv6"], [batch_size, 25000, 1, nkernels[2]], [1, dim_reduction, 1, 1]) + b["conv6"]), dropout_keep_prob)
	flattened = tf.reshape(tf.squeeze(h_conv6), [batch_size*25000, nkernels[2]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]
	return outputs, W, b, embed, [h_conv6]


def dilated_convolution_with_dilated_pooling_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, dim_reduction=10, nkernels=[128, 240, 50], hidden=125):

	# dim_reduction = 10
	# nkernels = [128, 240, 50]
	# hidden = 125
	embedding_size = 4
	filter_height = 5
	W = {
		"conv1": weight_variable([dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		"conv2": weight_variable([dim_reduction, 1, nkernels[0], nkernels[1]]),
		"conv3": weight_variable([dim_reduction, 1, nkernels[1], nkernels[2]], "W_conv3"),
		"conv4": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv5": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv6": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]], "W_conv6"),
		# "conv7": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"hid1": weight_variable([nkernels[2], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"conv2": bias_variable([nkernels[1]]),
		"conv3": bias_variable([nkernels[2]]),
		"conv4": bias_variable([nkernels[2]]),
		"conv5": bias_variable([nkernels[2]]),
		"conv6": bias_variable([nkernels[2]]),
		"conv7": bias_variable([nkernels[2]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)

	# add the convolutional layers
	h_conv1 = tf.nn.dropout(tf.nn.relu(_batch_norm(tf.nn.conv2d(embedded_image, W["conv1"], strides=[1, dim_reduction, 1, 1], padding='SAME') + b["conv1"], is_training, decay_rate, "conv1-norm")), dropout_keep_prob)
	h_conv2 = tf.nn.dropout(tf.nn.relu(_batch_norm(atrous_conv2d(h_conv1, W["conv2"], 3) + b["conv2"], is_training, decay_rate, "conv2-norm")), dropout_keep_prob)
	h_conv3 = tf.nn.dropout(tf.nn.relu(_batch_norm(atrous_conv2d(h_conv2, W["conv3"], 9) + b["conv3"], is_training, decay_rate, "conv3-norm")), dropout_keep_prob)
	h_conv4 = tf.nn.dropout(tf.nn.relu(_batch_norm(atrous_conv2d(h_conv3, W["conv4"], 27) + b["conv4"], is_training, decay_rate, "conv4-norm")), dropout_keep_prob)
	h_conv4_pooled = atrous_pool(h_conv4, filter_height, 27)
	h_conv5 = tf.nn.dropout(tf.nn.relu(_batch_norm(atrous_conv2d(h_conv4_pooled, W["conv5"], 81) + b["conv5"], is_training, decay_rate, "conv5-norm")), dropout_keep_prob)
	h_conv5_pooled = atrous_pool(h_conv5, filter_height, 81)
	h_conv6 = tf.nn.dropout(tf.nn.relu(_batch_norm(tf.nn.conv2d_transpose(h_conv5_pooled, W["conv6"], [batch_size, 25000, 1, nkernels[2]], [1, dim_reduction, 1, 1]) + b["conv6"], is_training, decay_rate, "conv6-norm")), dropout_keep_prob)
	flattened = tf.reshape(tf.squeeze(h_conv6), [batch_size*25000, nkernels[2]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]
	return outputs, W, b, embed, [h_conv6]

def convolution_7_layer_resizing(x, y_, dropout_keep_prob, batch_size, noutputs, dim_reduction=10, nkernels=[128, 240, 50], hidden=125):

	# dim_reduction = 10
	# nkernels = [128, 240, 50]
	# hidden = 125
	embedding_size = 4
	W = {
		"conv1": weight_variable([dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		"conv2": weight_variable([dim_reduction, 1, nkernels[0], nkernels[1]]),
		"conv3": weight_variable([dim_reduction, 1, nkernels[1], nkernels[2]], "W_conv3"),
		"conv4": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv5": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv6": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]], "W_conv6"),
		"conv7": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]], "W_conv7"),
		"hid1": weight_variable([nkernels[2], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"conv2": bias_variable([nkernels[1]]),
		"conv3": bias_variable([nkernels[2]]),
		"conv4": bias_variable([nkernels[2]]),
		"conv5": bias_variable([nkernels[2]]),
		"conv6": bias_variable([nkernels[2]]),
		"conv7": bias_variable([nkernels[2]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)

	# add the convolutional layers
	h_conv1 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(embedded_image, W["conv1"], strides=[1, dim_reduction, 1, 1], padding='SAME') + b["conv1"]), dropout_keep_prob)
	h_conv2 = tf.nn.dropout(tf.nn.relu(conv2d(h_conv1, W["conv2"]) + b["conv2"]), dropout_keep_prob)
	h_conv3 = tf.nn.dropout(tf.nn.relu(conv2d(h_conv2, W["conv3"]) + b["conv3"]), dropout_keep_prob)
	h_conv4 = tf.nn.dropout(tf.nn.relu(conv2d(h_conv3, W["conv4"]) + b["conv4"]), dropout_keep_prob)
	h_conv5 = tf.nn.dropout(tf.nn.relu(conv2d(h_conv4, W["conv5"]) + b["conv5"]), dropout_keep_prob)
	h_conv6 = tf.nn.dropout(tf.nn.relu(conv2d(h_conv5, W["conv6"]) + b["conv6"]), dropout_keep_prob)
	h_conv7 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d_transpose(h_conv6, W["conv7"], [batch_size, 25000, 1, nkernels[2]], [1, dim_reduction, 1, 1]) + b["conv7"]), dropout_keep_prob)
	flattened = tf.reshape(tf.squeeze(h_conv7), [batch_size*25000, nkernels[2]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]

	return outputs, W, b, embed, [h_conv7]

def convolution_7_layer_resizing_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, dim_reduction=10, nkernels=[128, 240, 50], hidden=125):

	# dim_reduction = 10
	# nkernels = [128, 240, 50]
	# hidden = 125
	embedding_size = 4
	W = {
		"conv1": weight_variable([dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		"conv2": weight_variable([dim_reduction, 1, nkernels[0], nkernels[1]]),
		"conv3": weight_variable([dim_reduction, 1, nkernels[1], nkernels[2]], "W_conv3"),
		"conv4": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv5": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv6": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]], "W_conv6"),
		"conv7": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]], "W_conv7"),
		"hid1": weight_variable([nkernels[2], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"conv2": bias_variable([nkernels[1]]),
		"conv3": bias_variable([nkernels[2]]),
		"conv4": bias_variable([nkernels[2]]),
		"conv5": bias_variable([nkernels[2]]),
		"conv6": bias_variable([nkernels[2]]),
		"conv7": bias_variable([nkernels[2]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)

	# add the convolutional layers
	h_conv1 = tf.nn.dropout(tf.nn.relu(_batch_norm(tf.nn.conv2d(embedded_image, W["conv1"], strides=[1, dim_reduction, 1, 1], padding='SAME') + b["conv1"], is_training, decay_rate, "conv1-norm")), dropout_keep_prob)
	h_conv2 = tf.nn.dropout(tf.nn.relu(_batch_norm(conv2d(h_conv1, W["conv2"]) + b["conv2"], is_training, decay_rate, "conv2-norm")), dropout_keep_prob)
	h_conv3 = tf.nn.dropout(tf.nn.relu(_batch_norm(conv2d(h_conv2, W["conv3"]) + b["conv3"], is_training, decay_rate, "conv3-norm")), dropout_keep_prob)
	h_conv4 = tf.nn.dropout(tf.nn.relu(_batch_norm(conv2d(h_conv3, W["conv4"]) + b["conv4"], is_training, decay_rate, "conv4-norm")), dropout_keep_prob)
	h_conv5 = tf.nn.dropout(tf.nn.relu(_batch_norm(conv2d(h_conv4, W["conv5"]) + b["conv5"], is_training, decay_rate, "conv5-norm")), dropout_keep_prob)
	h_conv6 = tf.nn.dropout(tf.nn.relu(_batch_norm(conv2d(h_conv5, W["conv6"]) + b["conv6"], is_training, decay_rate, "conv6-norm")), dropout_keep_prob)
	h_conv7 = tf.nn.dropout(tf.nn.relu(_batch_norm(tf.nn.conv2d_transpose(h_conv6, W["conv7"], [batch_size, 25000, 1, nkernels[2]], [1, dim_reduction, 1, 1]) + b["conv7"], is_training, decay_rate, "conv7-norm")), dropout_keep_prob)
	flattened = tf.reshape(tf.squeeze(h_conv7), [batch_size*25000, nkernels[2]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]

	return outputs, W, b, embed, [h_conv7]

def convolution_3_layer_resizing(x, y_, dropout_keep_prob, batch_size, noutputs, dim_reduction=10, nkernels=[128, 240, 50], hidden=125):

	# dim_reduction = 10
	# nkernels = [128, 240, 50]
	# hidden = 125
	embedding_size = 4
	W = {
		"conv1": weight_variable([dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		"conv2": weight_variable([dim_reduction, 1, nkernels[0], nkernels[1]], "W_conv2"),
		"conv3": weight_variable([dim_reduction, 1, nkernels[2], nkernels[1]], "W_conv3"),
		"hid1": weight_variable([nkernels[2], hidden], "W_hid1"),
		"hid2": weight_variable([hidden, noutputs], "W_hid2"),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"conv2": bias_variable([nkernels[1]]),
		"conv3": bias_variable([nkernels[2]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)

	# add the convolutional layers
	h_conv1 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(embedded_image, W["conv1"], strides=[1, dim_reduction, 1, 1], padding='SAME') + b["conv1"]), dropout_keep_prob)
	h_conv2 = tf.nn.dropout(tf.nn.relu(conv2d(h_conv1, W["conv2"]) + b["conv2"]), dropout_keep_prob)
	h_conv3 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d_transpose(h_conv2, W["conv3"], [batch_size, 25000, 1, nkernels[2]], [1, dim_reduction, 1, 1]) + b["conv3"]), dropout_keep_prob)
	flattened = tf.reshape(tf.squeeze(h_conv3), [batch_size*25000, nkernels[2]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]

	return outputs, W, b, embed, [h_conv3]

def convolution_3_layer_resizing_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, dim_reduction=10, nkernels=[128, 240, 50], hidden=125):

	# dim_reduction = 10
	# nkernels = [128, 240, 50]
	# hidden = 125
	embedding_size = 4
	W = {
		"conv1": weight_variable([dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		"conv2": weight_variable([dim_reduction, 1, nkernels[0], nkernels[1]], "W_conv2"),
		"conv3": weight_variable([dim_reduction, 1, nkernels[2], nkernels[1]], "W_conv3"),
		"hid1": weight_variable([nkernels[2], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"conv2": bias_variable([nkernels[1]]),
		"conv3": bias_variable([nkernels[2]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)

	# add the convolutional layers
	h_conv1 = tf.nn.dropout(tf.nn.relu(_batch_norm(tf.nn.conv2d(embedded_image, W["conv1"], strides=[1, dim_reduction, 1, 1], padding='SAME') + b["conv1"], is_training, decay_rate, "conv1-norm")), dropout_keep_prob)
	h_conv2 = tf.nn.dropout(tf.nn.relu(_batch_norm(conv2d(h_conv1, W["conv2"]) + b["conv2"], is_training, decay_rate, "conv2-norm")), dropout_keep_prob)
	h_conv3 = tf.nn.dropout(tf.nn.relu(_batch_norm(tf.nn.conv2d_transpose(h_conv2, W["conv3"], [batch_size, 25000, 1, nkernels[2]], [1, dim_reduction, 1, 1]) + b["conv3"], is_training, decay_rate, "conv3-norm")), dropout_keep_prob)
	flattened = tf.reshape(tf.squeeze(h_conv3), [batch_size*25000, nkernels[2]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]

	return outputs, W, b, embed, [h_conv3]


def convolution_1_layer(x, y_, dropout_keep_prob, batch_size, noutputs, dim_reduction=10, nkernels=[128, 240, 50], hidden=125):

	# dim_reduction = 10
	# nkernels = [128, 240, 50]
	# hidden = 125
	embedding_size = 4
	W = {
		"conv1": weight_variable([dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		"hid1": weight_variable([nkernels[0], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)

	# add the convolutional layers
	h_conv1 = tf.nn.dropout(tf.nn.relu(conv2d(embedded_image, W["conv1"]) + b["conv1"]), dropout_keep_prob)
	flattened = tf.reshape(tf.squeeze(h_conv1), [batch_size*25000, nkernels[0]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]

	return outputs, W, b, embed, [h_conv1]

def convolution_3_layer(x, y_, dropout_keep_prob, batch_size, noutputs, dim_reduction=10, nkernels=[128, 240, 50], hidden=125):
	# dim_reduction = 10
	# nkernels = [128, 240, 50]
	# hidden = 125
	embedding_size = 4
	W = {
		"conv1": weight_variable([dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		"conv2": weight_variable([dim_reduction, 1, nkernels[0], nkernels[1]], "W_conv2"),
		"conv3": weight_variable([dim_reduction, 1, nkernels[1], nkernels[2]], "W_conv3"),
		"hid1": weight_variable([nkernels[2], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"conv2": bias_variable([nkernels[1]]),
		"conv3": bias_variable([nkernels[2]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)

	# add the convolutional layers
	h_conv1 = tf.nn.dropout(tf.nn.relu(conv2d(embedded_image, W["conv1"])+ b["conv1"]), dropout_keep_prob)
	h_conv2 = tf.nn.dropout(tf.nn.relu(conv2d(h_conv1, W["conv2"]) + b["conv2"]), dropout_keep_prob)
	h_conv3 = tf.nn.dropout(tf.nn.relu(conv2d(h_conv2, W["conv3"]) + b["conv3"]), dropout_keep_prob)
	flattened = tf.reshape(tf.squeeze(h_conv3), [batch_size*25000, nkernels[2]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]

	return outputs, W, b, embed, [h_conv3]

def conv_bi_lstm(x, y_, dropout_keep_prob, batch_size, noutputs, dim_reduction=100, nkernels=[128, 20, 50], hidden=125):
	# dim_reduction = 100
	# nkernels = [128, 240, 50]
	# n_hidden = 20
	# hidden = 125
	n_hidden = nkernels[1]
	embedding_size = 4

	fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

	total_output_size = fw_cell.output_size + bw_cell.output_size

	W = {
		"conv1": weight_variable([dim_reduction, 1, embedding_size, nkernels[0]], "W_conv1"),
		#"conv3": weight_variable([dim_reduction, 1, nkernels[2], nkernels[0]]),
		"conv3": weight_variable([dim_reduction, 1, nkernels[2], total_output_size], "W_conv3"),
		"hid1": weight_variable([nkernels[2], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv1": bias_variable([nkernels[0]]),
		"conv3": bias_variable([nkernels[2]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	print 1, embed.get_shape()
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)
	print 2, embedded_image.get_shape()
	
	h_conv1 = tf.nn.relu(tf.nn.conv2d(embedded_image, W["conv1"], strides=[1, dim_reduction, 1, 1], padding='SAME') + b["conv1"])
	print 3, h_conv1.get_shape()
	h_conv1_sq = tf.squeeze(tf.nn.dropout(h_conv1, dropout_keep_prob), squeeze_dims=[2])
	print 4, h_conv1_sq.get_shape()
	#splitted = tf.unstack(h_conv1_sq, axis=1)
	seq_len = h_conv1_sq.get_shape()[1]
	print seq_len, batch_size
	seq_lens = tf.ones([batch_size], tf.int32)*seq_len
	print seq_lens
	outputs, _,= tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, h_conv1_sq, dtype=tf.float32, time_major=False, sequence_length=seq_lens)
	packed = tf.expand_dims(tf.concat(outputs, 2), 2)
	print 5, packed.get_shape()
	#concatenated = tf.nn.dropout(tf.concat(2, outputs), dropout_keep_prob)
	h_conv3 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d_transpose(packed, W["conv3"], [batch_size, 25000, 1, nkernels[2]], [1, dim_reduction, 1, 1]) + b["conv3"]), dropout_keep_prob)
	print 6, h_conv3.get_shape()
	# add the convolutional layers
	#h_conv1 = tf.nn.dropout(tf.nn.relu(conv2d(embedded_image, W["conv1"]) + b["conv1"]), .2)
	flattened = tf.reshape(tf.squeeze(h_conv3, squeeze_dims=[2]), [batch_size*25000, nkernels[2]])
	print 7, flattened.get_shape()
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	print 8, hid1.get_shape()
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]

	return outputs, W, b, embed, [h_conv3]


def ID_block_batchnorm(inp, W, b, dim_reduction, dropout_keep_prob, batch_size, nkernels, is_training, decay_rate, block_suffix):
	h_conv1 = tf.nn.conv2d(inp, W["conv1"], strides=[1, 2*dim_reduction, 1, 1], padding='SAME') + b["conv1"]
	h_conv1_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv1, is_training, decay_rate, "conv1-norm" + block_suffix)), dropout_keep_prob)
	h_conv2 = atrous_conv2d(h_conv1_norm, W["conv2"], 3) + b["conv2"]
	h_conv2_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv2, is_training, decay_rate, "conv2-norm" + block_suffix)), dropout_keep_prob)
	h_conv3 = atrous_conv2d(h_conv2_norm, W["conv3"], 9) + b["conv3"]
	h_conv3_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv3, is_training, decay_rate, "conv3-norm" + block_suffix)), dropout_keep_prob)
	h_conv4 = atrous_conv2d(h_conv3_norm, W["conv4"], 27) + b["conv4"]
	h_conv4_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv4, is_training, decay_rate, "conv4-norm" + block_suffix)), dropout_keep_prob)
	h_conv5 = tf.nn.conv2d_transpose(h_conv4_norm, W["conv5"], [batch_size, 25000, 1, nkernels[0]], [1, 2*dim_reduction, 1, 1]) + b["conv5"]
	h_conv5_norm = tf.nn.dropout(tf.nn.relu(_batch_norm(h_conv5, is_training, decay_rate, "conv5-norm" + block_suffix)), dropout_keep_prob)
	return h_conv5_norm

def ID_block(inp, W, b, dim_reduction, dropout_keep_prob, batch_size, nkernels):
	h_conv1 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(inp, W["conv1"], strides=[1, 2*dim_reduction, 1, 1], padding='SAME') + b["conv1"]), dropout_keep_prob)
	h_conv2 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv1, W["conv2"], 3) + b["conv2"]), dropout_keep_prob)
	h_conv3 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv2, W["conv3"], 9) + b["conv3"]), dropout_keep_prob)
	h_conv4 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv3, W["conv4"], 27) + b["conv4"]), dropout_keep_prob)
	h_conv5 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d_transpose(h_conv4, W["conv5"], [batch_size, 25000, 1, nkernels[0]], [1, 2*dim_reduction, 1, 1]) + b["conv5"]), dropout_keep_prob)
	return h_conv5

def get_block_outputs(block_activations, batch_size, nkernels, dropout_keep_prob, W, b):
	flattened = tf.reshape(tf.squeeze(block_activations), [batch_size*25000, nkernels[0]])
	hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]
	return outputs

def ID_CNN_model(x, y_, dropout_keep_prob, batch_size, noutputs, is_training=None, decay_rate=None, use_batchnorm=False, dim_reduction=10, nkernels=[50, 240, 80], hidden=125):
	# dim_reduction = 10
	# nkernels = [50, 240, 80]
	# hidden = 125
	embedding_size = 4
	W = {
		"conv0": weight_variable([2*dim_reduction, 1, embedding_size, nkernels[0]], "W_conv0"),
		"conv1": weight_variable([2*dim_reduction, 1, nkernels[0], nkernels[0]], "W_conv1"),
		"conv2": weight_variable([dim_reduction, 1, nkernels[0], nkernels[1]], "W_conv2"),
		"conv3": weight_variable([dim_reduction, 1, nkernels[1], nkernels[2]]),
		"conv4": weight_variable([dim_reduction, 1, nkernels[2], nkernels[2]]),
		"conv5": weight_variable([2*dim_reduction, 1, nkernels[0], nkernels[2]], "W_conv5"),
		"hid1": weight_variable([nkernels[0], hidden]),
		"hid2": weight_variable([hidden, noutputs]),
	}

	b = {
		"conv0": bias_variable([nkernels[0]]),
		"conv1": bias_variable([nkernels[0]]),
		"conv2": bias_variable([nkernels[1]]),
		"conv3": bias_variable([nkernels[2]]),
		"conv4": bias_variable([nkernels[2]]),
		"conv5": bias_variable([nkernels[0]]),
		"hid1": bias_variable([hidden]),
		"hid2": bias_variable([noutputs]),
	}

	embeddings = tf.Variable(tf.random_uniform([5, embedding_size]))
	embed = tf.nn.embedding_lookup(embeddings, x)
	# and a empty axis for the height
	embedded_image = tf.expand_dims(embed, 2)
	h_conv0 = tf.nn.dropout(tf.nn.relu(conv2d(embedded_image, W["conv0"]) + b["conv0"]), dropout_keep_prob)

	# add the convolutional layers
	if use_batchnorm:
		block_1 = ID_block_batchnorm(h_conv0, W, b, dim_reduction, dropout_keep_prob, batch_size, nkernels, is_training, decay_rate, "_block1")
		block_2 = ID_block_batchnorm(block_1, W, b, dim_reduction, dropout_keep_prob, batch_size, nkernels, is_training, decay_rate, "_block2")
		block_3 = ID_block_batchnorm(block_2, W, b, dim_reduction, dropout_keep_prob, batch_size, nkernels, is_training, decay_rate, "_block3")
	else:
		block_1 = ID_block(h_conv0, W, b, dim_reduction, dropout_keep_prob, batch_size, nkernels)
		block_2 = ID_block(block_1, W, b, dim_reduction, dropout_keep_prob, batch_size, nkernels)
		block_3 = ID_block(block_2, W, b, dim_reduction, dropout_keep_prob, batch_size, nkernels)

	block_1_outputs = get_block_outputs(block_1, batch_size, nkernels, dropout_keep_prob, W, b)
	block_2_outputs = get_block_outputs(block_2, batch_size, nkernels, dropout_keep_prob, W, b)
	block_3_outputs = get_block_outputs(block_3, batch_size, nkernels, dropout_keep_prob, W, b)

	# h_conv1 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(embedded_image, W["conv1"], strides=[1, 2*dim_reduction, 1, 1], padding='SAME') + b["conv1"]), dropout_keep_prob)
	# h_conv2 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv1, W["conv2"], 3) + b["conv2"]), dropout_keep_prob)
	# h_conv3 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv2, W["conv3"], 9) + b["conv3"]), dropout_keep_prob)
	# h_conv4 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv3, W["conv4"], 27) + b["conv4"]), dropout_keep_prob)
	# h_conv5 = tf.nn.dropout(tf.nn.relu(atrous_conv2d(h_conv4, W["conv5"], 81) + b["conv5"]), dropout_keep_prob)
	# h_conv6 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d_transpose(h_conv5, W["conv6"], [batch_size, 25000, 1, nkernels[2]], [1, 2*dim_reduction, 1, 1]) + b["conv6"]), dropout_keep_prob)
	
	# flattened = tf.reshape(tf.squeeze(block_3), [batch_size*25000, nkernels[0]])
	# hid1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, W["hid1"]) + b["hid1"]), dropout_keep_prob)
	# outputs = tf.matmul(hid1, W["hid2"]) + b["hid2"]
	
	return [block_1_outputs, block_2_outputs, block_3_outputs], W, b, embed, None

def get_model(model, x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, kw, nkernels, hidden):
	if model == "dilated":
		return dilated_convolution_model(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
	elif model == "dilated_normed":
		return dilated_convolution_model_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, kw, nkernels, hidden)
	elif model == "dilated_pooling":
		return  dilated_convolution_with_pooling(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
	elif model == "dilated_pooling_normed":
		return dilated_convolution_with_pooling_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, kw, nkernels, hidden)
	elif model == "dilated_dil_pooling":
		return  dilated_convolution_with_dilated_pooling(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
	elif model == "dilated_dil_pooling_normed":
		return dilated_convolution_with_dilated_pooling_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, kw, nkernels, hidden)
	elif model == "conv1":
		return convolution_1_layer(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
	elif model == "conv3":
		return convolution_3_layer(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
	elif model == "conv3_resizing":
		return convolution_3_layer_resizing(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
	elif model == "conv3_resizing_normed":
		return convolution_3_layer_resizing_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, kw, nkernels, hidden)
	elif model == "conv7_resizing":
		return convolution_7_layer_resizing(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
	elif model == "conv7_resizing_normed":
		return convolution_7_layer_resizing_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, kw, nkernels, hidden)
	elif model == "conv_bi_lstm":
		return conv_bi_lstm(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
	elif model == "id_cnn":
		return ID_CNN_model(x, y_, dropout_keep_prob, batch_size, noutputs, None, None, False, kw, nkernels, hidden)
	elif model == "id_cnn_normed":
		return ID_CNN_model(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, True, kw, nkernels, hidden)
	else:
		print("Invalid model: " + model)
		sys.exit()
	return None, None, None, None, None
