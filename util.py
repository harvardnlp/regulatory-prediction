import numpy as np
from scipy.sparse import csr_matrix
import gc
import sys
from sklearn.metrics import precision_recall_curve, auc
import random

def pr_auc_score(y_true, y_score):
	precision, recall, _ = precision_recall_curve(y_true, y_score)
	return auc(recall, precision, reorder=True), auc(precision, recall, reorder=True)

def get_consistent_filename(args, version=3):
	if version == 1:
		return "model_" + args.model + "_" + "epochs_" + str(args.epochs) + "_dropout_" + str(args.dropout) + "_rate_" + str(args.learning_rate) + "_batch_" + str(args.batch_size) + "_" + args.suffix
	if version == 2:
		return "model_" + args.model + "_epochs_" + str(args.epochs) + "_dropout_" + str(args.dropout) + "_rate_" + str(args.learning_rate) + "_batch_" + str(args.batch_size) + "_weight_" + str(args.pos_weight) + "_" + args.suffix
	if version == 3:
		return "model_" + args.model + "_epochs_" + str(args.epochs) + "_dropout_" + str(args.dropout) + "_rate_" + str(args.learning_rate) + "_batch_" + str(args.batch_size) + "_weight_" + str(args.pos_weight) + "_batchdecay_" + str(args.batch_decay) + "_" + args.suffix
	if version == 4:
		return "m_" + args.model + "_e_" + str(args.epochs) + "_d_" + str(args.dropout) + "_r_" + str(args.learning_rate) + "_b_" + str(args.batch_size) + "_w_" + str(args.pos_weight) + "_bd_" + str(args.batch_decay) +  "_kw_" + str(args.kw) +  "_size_" + str(args.w1) + "_" + str(args.w2) + "_" + str(args.w3) + "_hid_" + str(args.hidden) + "_" + args.suffix
class dotdict(dict):
	"""dot.notation access to dictionary attributes. 
	From http://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
 
# This function assumes that the training data are in files: chr1_batched.npz, chr2_batched.npz, ...
def training_minibatcher(batch_size=4,file_extension="_batched_nomultimap.npz", base_path="/n/rush_lab/data/chromatin-features/chromatin-nn/", debug=False, small_dataset=False, start=0):
	chromosomes = map(str, range(1, 23)) + ["X", "Y"]
	# If only a few are wanted, just look at the X and Y chromosomes
	if small_dataset:
		chromosomes = ["Y", "X"]
	random.shuffle(chromosomes)
	for chromosome in chromosomes:
		file = base_path + "hg19/chr" + chromosome + file_extension
		if debug:
			print("Loading the file")
			sys.stdout.flush()
		file_loaded = np.load(file)
		if debug:
			print("Loading the inputs")
			sys.stdout.flush()
		train_inputs = file_loaded["train_inputs"]
		if debug:
			print("Number of train_inputs", train_inputs.shape[0])
		if debug:
			print("Loading the outputs")
			sys.stdout.flush()
		train_outputs = file_loaded["train_outputs"]
		assert(train_inputs.shape[0] == train_outputs.shape[0])
		num_batches = train_inputs.shape[0]
		del file_loaded
		if debug:
			print("Running garbage collection")
			sys.stdout.flush()
		gc.collect()
		if debug:
			print("Yielding results")
			sys.stdout.flush()
		for i in xrange(start*batch_size, num_batches, batch_size):
			yield train_inputs[i:i+batch_size], train_outputs[i:i+batch_size]
			train_inputs[i:i+batch_size] = 0
			train_outputs[i:i+batch_size] = 0
		del train_inputs
		del train_outputs
		gc.collect()

def test_minibatcher(batch_size=4,file_extension="_batched_nomultimap.npz", base_path="/n/rush_lab/data/chromatin-features/chromatin-nn/", debug=False, small_dataset=False):
	chromosomes = map(str, range(1, 23)) + ["X", "Y"]
	# If only a few are wanted, just look at the X and Y chromosomes
	# if small_dataset:
	# 	chromosomes = ["Y", "X"]
	for chromosome in chromosomes:
		file = base_path + "hg19/chr" + chromosome + file_extension
		if debug:
			print("Loading the file")
			sys.stdout.flush()
		file_loaded = np.load(file)
		if debug:
			print("Loading the inputs")
			sys.stdout.flush()
		test_inputs = file_loaded["test_inputs"]
		if debug:
			print("Loading the outputs")
			sys.stdout.flush()
		test_outputs = file_loaded["test_outputs"]
		assert(test_inputs.shape[0] == test_outputs.shape[0])
		num_batches = test_inputs.shape[0]
		# If I only want a few, only show 5 percent of the full number of batches.
		if small_dataset:
			num_batches = int(num_batches*.05)
		del file_loaded
		if debug:
			print("Running garbage collection")
			sys.stdout.flush()
		gc.collect()
		if debug:
			print("Yielding results")
			sys.stdout.flush()
		for i in xrange(0, num_batches, batch_size):
			yield test_inputs[i:i+batch_size], test_outputs[i:i+batch_size]
			test_inputs[i:i+batch_size] = 0
			test_outputs[i:i+batch_size] = 0
		del test_inputs
		del test_outputs
		gc.collect()

def valid_minibatcher(batch_size=4,file_extension="_batched_nomultimap.npz", base_path="/n/rush_lab/data/chromatin-features/chromatin-nn/", debug=False, small_dataset=False):
	chromosomes = map(str, range(1, 23)) + ["X", "Y"]
	# If only a few are wanted, just look at the X and Y chromosomes
	for chromosome in chromosomes:
		file = base_path + "hg19/chr" + chromosome + file_extension
		if debug:
			print("Loading the file")
			sys.stdout.flush()
		file_loaded = np.load(file)
		if debug:
			print("Loading the inputs")
			sys.stdout.flush()
		valid_inputs = file_loaded["valid_inputs"]
		if debug:
			print("Loading the outputs")
			sys.stdout.flush()
		valid_outputs = file_loaded["valid_outputs"]
		assert(valid_inputs.shape[0] == valid_outputs.shape[0])
		num_batches = valid_inputs.shape[0]
		# If I only want a few, only show 10 percent of the full number of batches.
		if small_dataset:
			num_batches = int(num_batches*.10)
		del file_loaded
		if debug:
			print("Running garbage collection")
			sys.stdout.flush()
		gc.collect()
		if debug:
			print("Yielding results")
			sys.stdout.flush()
		for i in xrange(0, num_batches, batch_size):
			yield valid_inputs[i:i+batch_size], valid_outputs[i:i+batch_size]
			valid_inputs[i:i+batch_size] = 0
			valid_outputs[i:i+batch_size] = 0
		del valid_inputs
		del valid_outputs
		gc.collect()


# def get_model(model, x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate):
# 	if model == "dilated":
# 		return dilated_convolution_model(x, y_, dropout_keep_prob, batch_size, noutputs)
# 	elif model == "dilated_normed":
# 		return dilated_convolution_model_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate)
# 	elif model == "dilated_pooling":
# 		return dilated_convolution_with_pooling(x, y_, dropout_keep_prob, batch_size, noutputs)
# 	elif model == "dilated_pooling_normed":
# 		return dilated_convolution_with_pooling_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate)
# 	elif model == "dilated_dil_pooling":
# 		return dilated_convolution_with_dilated_pooling(x, y_, dropout_keep_prob, batch_size, noutputs)
# 	elif model == "dilated_dil_pooling_normed":
# 		return dilated_convolution_with_dilated_pooling_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate)
# 	elif model == "conv1":
# 		return convolution_1_layer(x, y_, dropout_keep_prob, batch_size, noutputs)
# 	elif model == "conv3":
# 		return convolution_3_layer(x, y_, dropout_keep_prob, batch_size, noutputs)
# 	elif model == "conv3_resizing":
# 		return convolution_3_layer_resizing(x, y_, dropout_keep_prob, batch_size, noutputs)
# 	elif model == "conv3_resizing_normed":
# 		return convolution_3_layer_resizing_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate)
# 	elif model == "conv7_resizing":
# 		return convolution_7_layer_resizing(x, y_, dropout_keep_prob, batch_size, noutputs)
# 	elif model == "conv7_resizing_normed":
# 		return convolution_7_layer_resizing_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate)
# 	elif model == "conv_bi_lstm":
# 		return conv_bi_lstm(x, y_, dropout_keep_prob, batch_size, noutputs)
# 	elif model == "id_cnn":
# 		return ID_CNN_model(x, y_, dropout_keep_prob, batch_size, noutputs)
# 	elif model == "id_cnn_normed":
# 		return ID_CNN_model(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, use_batchnorm=True)
# 	else:
# 		print("Invalid model: " + model)
# 		sys.exit()
# 	return None, None, None, None, None


# def get_model(model, x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, kw, nkernels, hidden):
# 	if model == "dilated":
# 		return dilated_convolution_model(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
# 	elif model == "dilated_normed":
# 		return dilated_convolution_model_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, kw, nkernels, hidden)
# 	elif model == "dilated_pooling":
# 		return  dilated_convolution_with_pooling(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
# 	elif model == "dilated_pooling_normed":
# 		return dilated_convolution_with_pooling_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, kw, nkernels, hidden)
# 	elif model == "dilated_dil_pooling":
# 		return  dilated_convolution_with_dilated_pooling(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
# 	elif model == "dilated_dil_pooling_normed":
# 		return dilated_convolution_with_dilated_pooling_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, kw, nkernels, hidden)
# 	elif model == "conv1":
# 		return convolution_1_layer(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
# 	elif model == "conv3":
# 		return convolution_3_layer(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
# 	elif model == "conv3_resizing":
# 		return convolution_3_layer_resizing(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
# 	elif model == "conv3_resizing_normed":
# 		return convolution_3_layer_resizing_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, kw, nkernels, hidden)
# 	elif model == "conv7_resizing":
# 		return convolution_7_layer_resizing(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
# 	elif model == "conv7_resizing_normed":
# 		return convolution_7_layer_resizing_batchnorm(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, kw, nkernels, hidden)
# 	elif model == "conv_bi_lstm":
# 		return conv_bi_lstm(x, y_, dropout_keep_prob, batch_size, noutputs, kw, nkernels, hidden)
# 	elif model == "id_cnn":
# 		return ID_CNN_model(x, y_, dropout_keep_prob, batch_size, noutputs, None, None, False, kw, nkernels, hidden)
# 	elif model == "id_cnn_normed":
# 		return ID_CNN_model(x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, True, kw, nkernels, hidden)
# 	else:
# 		print("Invalid model: " + model)
# 		sys.exit()
# 	return None, None, None, None, None
