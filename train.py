import numpy as np
import tensorflow as tf
import sys
import gc
import time
from util import training_minibatcher, get_consistent_filename
from models import get_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-bp", "--base_path", 
	help="The base path of the directory", 
	default="/n/rush_lab/data/chromatin-features/chromatin-nn/")
parser.add_argument("-ckpt_bp", "--ckpt_base_path",
       	help="The base path of the large ckpt files directory",
       	default="/n/regal/rush_lab/ankitgupta/chromatin-nn/ckpt/")
parser.add_argument("-summary_bp", "--summary_base_path",
       	help="The base path of the large summary files directory",
       	default="/n/regal/rush_lab/ankitgupta/chromatin-nn/summary_logs/")
parser.add_argument("-m", "--model", help="The model to run", 
	default="dilated")
parser.add_argument("-b", "--batch_size", type=int, help="The batch size", default=8)
parser.add_argument("--num_outputs", type=int, help="Number of outputs", default=919)
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=1)
parser.add_argument("-d", "--dropout", type=float, help="Dropout keep prob", default=.75)
parser.add_argument("-l", "--learning_rate", type=float, help="Dropout keep prob", default=.001)
parser.add_argument("-w", "--pos_weight", type=float, help="Weight to assign positive class", default=1.0)
parser.add_argument("-bd", "--batch_decay", type=float, help="Batch normalization decay rate", default=.9)
parser.add_argument("-v", "--version", type=int, help="Naming version", default=4)
parser.add_argument("-kw", "--kw", type=int, help="Kernel width", default=10)
parser.add_argument("-w1", "--w1", type=int, help="Weight shape 1", default=128)
parser.add_argument("-w2", "--w2", type=int, help="Weight shape 2", default=240)
parser.add_argument("-w3", "--w3", type=int, help="Weight shape 3", default=50)
parser.add_argument("-hid", "--hidden", type=int, help="Hidden layer size", default=125)
parser.add_argument("suffix",
	help="String to append to name in checkpoints (make this unique or it will overwrite)")
args = parser.parse_args()

model = args.model
base_path = args.base_path
batch_size = args.batch_size
noutputs = args.num_outputs
decay_rate = args.batch_decay

print(args)

x = tf.placeholder(tf.int32, shape=[None, 25000])
y_ = tf.placeholder(tf.float32, shape=[None, noutputs])
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
is_training = tf.placeholder(tf.bool, name="is_training")

print("Making the model")
sys.stdout.flush()

outputs, W,b, embed, activations = get_model(model, x, y_, dropout_keep_prob, batch_size, noutputs, is_training, decay_rate, args.kw, [args.w1, args.w2, args.w3], args.hidden)

cross_entropy = None
if isinstance(outputs, list):
	num_outputs = len(outputs)
	cross_entropy = tf.divide(tf.add_n([tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_, output, args.pos_weight)) for output in outputs]), num_outputs)
else:
	cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_, outputs, args.pos_weight))
train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(cross_entropy)
tf.summary.scalar("cross_entropy", cross_entropy)
merged = tf.summary.merge_all()
init_op = tf.initialize_all_variables()
saver = tf.train.Saver()
config = tf.ConfigProto()

ckpt_name = args.ckpt_base_path + get_consistent_filename(args, version=args.version) + ".ckpt"
print "Checkpoint file name will be", ckpt_name
sys.stdout.flush()

with tf.Session(config=config) as sess:
	train_writer = tf.summary.FileWriter(args.summary_base_path + args.suffix + "/" + get_consistent_filename(args, version=args.version), sess.graph)

	sess.run(init_op)
	print("Beginning the forward pass")
	sys.stdout.flush()

	total_counter = 0
	for epoch in xrange(args.epochs):
		counter = 0 
		for inps, outs in training_minibatcher(batch_size, small_dataset=False, base_path=base_path):

			# If the batch is too small, move on (only happens to the last one)
			if inps.shape[0] != batch_size:
				continue
			got_batch = time.time()

			# Get the outputs in a good format
			for elem in range(len(outs)):
				outs[elem] = outs[elem].toarray()
			outs = np.vstack(outs)
			configured_batch = time.time()

			# Update the parameters
                        sess.run(train_step, feed_dict={x:inps, y_:outs, dropout_keep_prob:args.dropout, is_training:True})
			ran_update = time.time()
			sys.stdout.flush()

			# Occasionally, print some stats, and save a checkpoint of the model
			if counter % 25 == 0:
				summary, loss_val = sess.run([merged, cross_entropy], feed_dict={x:inps, y_:outs, dropout_keep_prob:args.dropout, is_training:True})
				print "Epoch:", epoch, "Batch:", counter, "Loss: ", loss_val
				print "Configuring batch Time (sec):",  configured_batch - got_batch, "Running update Time (sec):", ran_update - configured_batch
				train_writer.add_summary(summary, total_counter)
				sys.stdout.flush()
				save_path = saver.save(sess, ckpt_name)
				#print("Model saved in file: %s" % save_path)
				del inps
				del outs
				gc.collect()
			counter += 1
			total_counter += 1
		# At the end, make sure to save the model
		save_path = saver.save(sess, ckpt_name)

	


