from graphcnn.helper import *
from graphcnn.network import *
from graphcnn.layers import *
from sklearn.model_selection import KFold
import numpy as np
import tensorflow.compat.v1 as tf
import glob
import time
#import datetime
import os
from tensorflow.python.training import queue_runner
import logging
import time
logging.getLogger('tensorflow').disabled = True

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# This function is used to create tf.cond compatible tf.train.batch alternative
def _make_batch_queue(input, capacity, num_threads=1):
    queue = tf.PaddingFIFOQueue(capacity=capacity, dtypes=[s.dtype for s in input], shapes=[s.get_shape() for s in input])
    tf.summary.scalar("fraction_of_%d_full" % capacity,
           tf.cast(queue.size(), tf.float32) *
           (1. / capacity))
    enqueue_ops = [queue.enqueue(input)]*num_threads
    queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))
    return queue

@tf.function
def writeTensor(X, fn):
    X = tf.strings.as_string(tf.squeeze(X))
    tf.io.write_file(fn, X)

# This class is responsible for setting up and running experiments
# Also provides helper functions related to experiments (e.g. get accuracy)
class GraphCNNExperiment(object):
    def __init__(self, dataset_name, model_name, net_constructor):
        # Initialize all defaults
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_iterations = 200
        self.iterations_per_test = 50
        self.display_iter = 50
        self.snapshot_iter = 25
        self.train_batch_size = 0
        self.test_batch_size = 0
        self.crop_if_possible = False
        self.debug = False
        self.starter_learning_rate = 0.1
        self.learning_rate_exp = 0.1
        self.learning_rate_step = 1000
        self.reports = {}
        self.silent = False
        self.optimizer = 'momentum'
        
        self.train_sz = None
        self.full_sz = None
        
        self.net_constructor = net_constructor
        self.net = GraphCNNNetwork()
        self.net_desc = GraphCNNNetworkDescription()
        tf.compat.v1.reset_default_graph()
        
    # print_ext can be disabled through the silent flag
    def print_ext(self, *args):
        if self.silent == False:
            print_ext(*args)
            
    # Will retrieve the value stored as the maximum test accuracy on a trained network
    # SHOULD ONLY BE USED IF test_batch_size == ALL TEST SAMPLES
    def get_max_accuracy(self):
        tf.reset_default_graph()
        with tf.compat.v1.variable_scope('loss') as scope:
            max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            max_it = self.load_model(sess, saver)
            return sess.run(max_acc_test), max_it
        
    # Run all folds in a CV and calculate mean/std
    def run_kfold_experiments(self, is_regression, no_folds=10):
        acc = []
        
        self.net_constructor.create_network(self.net_desc, [])
        desc = self.net_desc.get_description()
        self.print_ext('Running CV for:', desc)
        start_time = time.time()
        for i in range(no_folds):
            tf.compat.v1.reset_default_graph()
            self.set_kfold(no_folds=no_folds, fold_id=i)
            cur_max, max_it = self.run(is_regression)
            self.print_ext('Fold %d max accuracy: %g at %d' % (i, cur_max, max_it))
            acc.append(cur_max)
        acc = np.array(acc)
        mean_acc= np.mean(acc)*100
        std_acc = np.std(acc)*100
        self.print_ext('Result is: %.2f (+- %.2f)' % (mean_acc, std_acc))
        
        verify_dir_exists('./results/')
        with open('./results/%s.txt' % self.dataset_name, 'a+') as file:
            file.write('%s\t%s\t%d-fold\t%d seconds\t%.2f (+- %.2f)\n' % (str(datetime.now()), desc, no_folds, time.time()-start_time, mean_acc, std_acc))
        return mean_acc, std_acc
        
    # Prepares samples for experiment, accepts a list (vertices, adjacency, labels) where:
    # vertices = list of NxC matrices where C is the same over all samples, N can be different between samples
    # adjacency = list of NxLxN tensors containing L NxN adjacency matrices of the given samples
    # labels = list of sample labels
    # len(vertices) == len(adjacency) == len(labels)
    def preprocess_data(self, dataset, is_regression):
        self.graph_size = np.array([s.shape[0] for s in dataset[0]]).astype(np.int64)
        
        self.largest_graph = max(self.graph_size)
        print("Graph size: {}".format(self.largest_graph))
        self.print_ext('Padding samples')
        self.graph_vertices = []
        self.graph_adjacency = []
        print("Vertices size: {}".format(dataset[0].shape))
        print("Adjacency size: {}".format(dataset[1].shape))
        print("Labels size: {}".format(dataset[2].shape))

        self.print_ext('Stacking samples')
        self.graph_vertices = dataset[0]
        self.graph_adjacency = dataset[1]
        self.graph_labels = dataset[2].astype(np.float32) if is_regression else dataset[2].astype(np.int64)
        self.current_V = dataset[0]
        self.current_A = dataset[1]
        self.labels = dataset[2].astype(np.float32) if is_regression else dataset[2].astype(np.int64)
        
        self.train_sz = int(dataset[3])
        self.full_sz = int(dataset[4])
        self.num_test_batches = int((self.full_sz - self.train_sz) / self.test_batch_size) + 1 if self.test_batch_size > 0 else 1
        
        self.no_samples = self.graph_labels.shape[0]
        
        single_sample = [self.graph_vertices, self.graph_adjacency, self.graph_labels, self.graph_size]
        
    # Create CV information
    def set_kfold(self, no_folds = 10, fold_id = 0):
        self.fold_id = fold_id
        self.train_idx, self.test_idx = np.array(list(range(0, self.train_sz))), np.array(list(range(self.train_sz, self.full_sz)))
        self.no_samples_train = self.train_idx.shape[0]
        self.no_samples_test = self.test_idx.shape[0]
        self.print_ext('Data ready. no_samples_train:', self.no_samples_train, 'no_samples_test:', self.no_samples_test)
        
        if self.train_batch_size == 0:
            self.train_batch_size = self.no_samples_train
        if self.test_batch_size == 0:
            self.test_batch_size = self.no_samples_test
        self.train_batch_size = min(self.train_batch_size, self.no_samples_train)
        self.test_batch_size = min(self.test_batch_size, self.no_samples_test)
        
    # This function is cropped before batch
    # Slice each sample to improve performance
    def crop_single_sample(self, single_sample):
        vertices = tf.slice(single_sample[0], np.array([0, 0], dtype=np.int64), tf.cast(tf.stack([single_sample[3], -1]), tf.int64))
        vertices.set_shape([None, self.graph_vertices.shape[2]])
        adjacency = tf.slice(tf.expand_dims(single_sample[1], 0), np.array([0, 0, 0], dtype=np.int64), tf.cast(tf.stack([single_sample[3], -1, single_sample[3]]), tf.int64))
        adjacency.set_shape([None, self.graph_adjacency.shape[2], None])
        
        # V, A, labels, mask
        return [vertices, adjacency, single_sample[2], tf.expand_dims(tf.ones(tf.slice(tf.shape(vertices), [0], [1])), axis=-1)]
        
    def create_input_variable(self, input):
        for i in range(len(input)):
            print(input[i].shape)
            placeholder = tf.compat.v1.placeholder(tf.as_dtype(input[i].dtype), shape=input[i].shape)
            var = tf.Variable(placeholder, trainable=False, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])
            self.variable_initialization[placeholder] = input[i]
            input[i] = var
        for i in input:
            print(i.shape)
        return input
    # Create input_producers and batch queues
    def create_data(self):
        with tf.device("/CPU:0"):
            with tf.compat.v1.variable_scope('input') as scope:
                # Create the training queue
                with tf.compat.v1.variable_scope('train_data') as scope:
                    self.print_ext('Creating training Tensorflow Tensors')
                    
                    # Create tensor with all training samples
                    training_samples = [self.graph_vertices, self.graph_adjacency, self.graph_labels, self.graph_size]
                    training_samples = [s[self.train_idx, ...] for s in training_samples]
                    
                    if self.crop_if_possible == False:
                        training_samples[3] = get_node_mask(training_samples[3], max_size=self.graph_vertices.shape[1])
                        
                    # Create tf.constants
                    training_samples = self.create_input_variable(training_samples)
                    
                    # Slice first dimension to obtain samples
                    single_sample = tf.train.slice_input_producer(training_samples, shuffle=True, capacity=self.train_batch_size)
                    
                    # Cropping samples improves performance but is not required
                    if self.crop_if_possible:
                        self.print_ext('Cropping smaller graphs')
                        single_sample = self.crop_single_sample(single_sample)
                    
                    # creates training batch queue
                    train_queue = _make_batch_queue(single_sample, capacity=self.train_batch_size*2, num_threads=6)

                # Create the test queue
                with tf.compat.v1.variable_scope('test_data') as scope:
                    self.print_ext('Creating test Tensorflow Tensors')
                    
                    # Create tensor with all test samples
                    test_samples = [self.graph_vertices, self.graph_adjacency, self.graph_labels, self.graph_size]
                    test_samples = [s[self.test_idx, ...] for s in test_samples]
                    
                    # If using mini-batch we will need a queue 
                    if self.test_batch_size != self.no_samples_test:
                        if self.crop_if_possible == False:
                            test_samples[3] = get_node_mask(test_samples[3], max_size=self.graph_vertices.shape[1])
                        test_samples = self.create_input_variable(test_samples)
                        
                        single_sample = tf.train.slice_input_producer(test_samples, shuffle=True, capacity=self.test_batch_size)
                        if self.crop_if_possible:
                            single_sample = self.crop_single_sample(single_sample)
                            
                        test_queue = _make_batch_queue(single_sample, capacity=self.test_batch_size*2, num_threads=1)
                        
                    # If using full-batch no need for queues
                    else:
                        test_samples[3] = get_node_mask(test_samples[3], max_size=self.graph_vertices.shape[1])
                        test_samples = self.create_input_variable(test_samples)
                        
                # obtain batch depending on is_training and if test is a queue
                if self.test_batch_size == self.no_samples_test:
                    return tf.cond(self.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_samples)
                return tf.cond(self.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_queue.dequeue_many(self.test_batch_size))
     
    # Function called with the output of the Graph-CNN model
    # Should add the loss to the 'losses' collection and add any summaries needed (e.g. accuracy) 
    def create_loss_function(self, is_regression):
        with tf.compat.v1.variable_scope('loss') as scope:
            self.print_ext('Creating loss function and summaries')
            if is_regression:
                prediction = tf.squeeze(tf.cast(self.net.current_V, tf.float32))
                labels = tf.squeeze(self.net.labels)
                
                mse = tf.losses.mean_squared_error(labels, prediction, scope='loss')
                
                total_error = tf.reduce_sum(tf.square(tf.subtract(labels, tf.reduce_mean(labels))))
                unexplained_error = tf.reduce_sum(tf.square(tf.subtract(labels, prediction)))
                R_squared = tf.subtract(float(1), tf.div(unexplained_error, total_error))
                
                tf.add_to_collection('losses', mse)
                tf.summary.scalar('loss', mse)

                self.max_r2_train = tf.Variable(tf.zeros([]), name="max_r2_train")
                self.max_r2_test = tf.Variable(tf.zeros([]), name="max_r2_test")
                
                max_r2 = tf.cond(self.net.is_training, lambda: tf.assign(self.max_r2_train, tf.maximum(self.max_r2_train, R_squared)), lambda: tf.assign(self.max_r2_test, tf.maximum(self.max_r2_test, R_squared)))
                tf.summary.scalar('max_r2', max_r2)
                tf.summary.scalar('accuracy', R_squared)

                self.reports['r2'] = R_squared
                self.reports['max r2'] = max_r2
                self.reports['mse'] = mse
            else:

                cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.net.current_V, labels=self.net.labels))

                correct_prediction = tf.cast(tf.equal(tf.argmax(self.net.current_V, 1), self.net.labels), tf.float32)
                accuracy = tf.reduce_mean(correct_prediction)

                # we have 2 variables that will keep track of the best accuracy obtained in training/testing batch
                # SHOULD ONLY BE USED IF test_batch_size == ALL TEST SAMPLES
                
                self.max_acc_train = tf.Variable(tf.zeros([]), name="max_acc_train")
                self.max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
                max_acc = tf.cond(self.net.is_training, lambda: tf.assign(self.max_acc_train, tf.maximum(self.max_acc_train, accuracy)), lambda: tf.assign(self.max_acc_test, tf.maximum(self.max_acc_test, accuracy)))

                tf.add_to_collection('losses', cross_entropy)
                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar('max_accuracy', max_acc)
                tf.summary.scalar('cross_entropy', cross_entropy)
                
                self.reports['accuracy'] = accuracy
                self.reports['max acc.'] = max_acc
                self.reports['cross_entropy'] = cross_entropy
        
    # check if the model has a saved iteration and return the latest iteration step
    def check_model_iteration(self):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None:
            return 0
        return int(latest[len(self.snapshot_path + 'model-'):])
        
    # load_model if any checkpoint exist
    def load_model(self, sess, saver, ):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None:
            return 0
        saver.restore(sess, latest)
        i = int(latest[len(self.snapshot_path + 'model-'):])
        self.print_ext("Model restored at %d." % i)
        return i
        
    def save_model(self, sess, saver, i):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None or i != int(latest[len(self.snapshot_path + 'model-'):]):
            self.print_ext('Saving model at %d' % i)
            verify_dir_exists(self.snapshot_path)
            result = saver.save(sess, self.snapshot_path + 'model', global_step=i)
            self.print_ext('Model saved to %s' % result)
      
    # Create graph (input, network, loss)
    # Handle checkpoints
    # Report summaries if silent == false
    # start/end threads
    def run(self, is_regression):
        with tf.device('/gpu:0'):
            self.variable_initialization = {}

            self.print_ext('Training model "%s"!' % self.model_name)
            if hasattr(self, 'fold_id') and self.fold_id:
                self.snapshot_path = '/atlas/u/dgrosz/mapillary-dhs/code/graph-cnn/src/snapshots/%s/%s/' % (self.dataset_name, self.model_name + '_fold%d' % self.fold_id)
                self.test_summary_path = '/atlas/u/dgrosz/mapillary-dhs/code/graph-cnn/src/summary/%s/test/%s_fold%d' %(self.dataset_name, self.model_name, self.fold_id)
                self.train_summary_path = '/atlas/u/dgrosz/mapillary-dhs/code/graph-cnn/src/summary/%s/train/%s_fold%d' %(self.dataset_name, self.model_name, self.fold_id)
            else:
                self.snapshot_path = '/atlas/u/dgrosz/mapillary-dhs/code/graph-cnn/src/snapshots/%s/%s/' % (self.dataset_name, self.model_name)
                self.test_summary_path = '/atlas/u/dgrosz/mapillary-dhs/code/graph-cnn/src/summary/%s/test/%s' %(self.dataset_name, self.model_name)
                self.train_summary_path = '/atlas/u/dgrosz/mapillary-dhs/code/graph-cnn/src/summary/%s/train/%s' %(self.dataset_name, self.model_name)
            if self.debug:
                i = 0
            else:
                i = self.check_model_iteration()
            if i < self.num_iterations:
                self.print_ext('Creating training network')

                self.net.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
                self.net.global_step = tf.Variable(0,name='global_step',trainable=False)


                input = self.create_data()
                self.net_constructor.create_network(self.net, input)
                self.create_loss_function(is_regression)

                self.print_ext('Preparing training')
                loss = tf.add_n(tf.get_collection('losses'))
                if len(tf.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)) > 0:
                    loss += tf.add_n(tf.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))

                update_ops = tf.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)



                with tf.control_dependencies(update_ops):
                    if self.optimizer == 'adam':
                        train_step = tf.train.AdamOptimizer().minimize(loss, global_step=self.net.global_step)
                    else:
                        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.net.global_step, self.learning_rate_step, self.learning_rate_exp, staircase=True)
                        train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(loss, global_step=self.net.global_step)
                        self.reports['lr'] = self.learning_rate
                        tf.summary.scalar('learning_rate', self.learning_rate)

                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer(), self.variable_initialization)

                    if self.debug == False:
                        saver = tf.train.Saver()
                        self.load_model(sess, saver)

                        self.print_ext('Starting summaries')
                        test_writer = tf.summary.FileWriter(self.test_summary_path, sess.graph)
                        train_writer = tf.summary.FileWriter(self.train_summary_path, sess.graph)

                    summary_merged = tf.summary.merge_all()

                    self.print_ext('Starting threads')
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    self.print_ext('Starting training. train_batch_size:', self.train_batch_size, 'test_batch_size:', self.test_batch_size)
                    wasKeyboardInterrupt = False
                    try:
                        total_training = 0.0
                        total_testing = 0.0
                        start_at = time.time()
                        last_summary = time.time()
                        while i < self.num_iterations:
                            if i % self.snapshot_iter == 0 and self.debug == False:
                                self.save_model(sess, saver, i)
                                print("Saving...")
                            if i % self.iterations_per_test == 0:
                                start_temp = time.time()
                                summary, reports = sess.run([summary_merged, self.reports], feed_dict={self.net.is_training:0})
                                total_testing += time.time() - start_temp
                                self.print_ext('Test Step %d Finished' % i)
                                for key, value in reports.items():
                                    self.print_ext('Test Step %d "%s" = ' % (i, key), value)
                                if self.debug == False:
                                    test_writer.add_summary(summary, i)

                            start_temp = time.time()
                            summary, _, reports = sess.run([summary_merged, train_step, self.reports], feed_dict={self.net.is_training:1})
                            total_training += time.time() - start_temp
                            i += 1
                            if ((i-1) % self.display_iter) == 0:
                                if self.debug == False:
                                    train_writer.add_summary(summary, i-1)
                                total = time.time() - start_at
                                self.print_ext('Training Step %d Finished Timing (Training: %g, Test: %g) after %g seconds' % (i-1, total_training/total, total_testing/total, time.time()-last_summary)) 
                                for key, value in reports.items():
                                    self.print_ext('Training Step %d "%s" = ' % (i-1, key), value)
# BETA: Uncomment the following code to save predictions
#                                 print("Saving predictions...")


#                                     with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess2:

#                                         sess2.run(tf.global_variables_initializer())
#                                         sess2.run(tf.local_variables_initializer(), self.variable_initialization)
#                                         correct_prediction = tf.cast(tf.equal(tf.argmax(self.net.current_V, 1), self.net.labels), tf.float32)
#                                         accuracy = tf.reduce_mean(correct_prediction)
#                                         prediction_save = tf.squeeze(tf.argmax(self.net.current_V,1))
#                                         prediction2_save = tf.squeeze(tf.argmax(self.net.current_V,0))
#                                         correct = tf.squeeze(self.net.labels)
#                                         def save():
#                                             #sess2.run(placeholder_fix, feed_dict={placeholder_pred: self.net.current_V})
#                                             def save_train():
#                                                 np.savetxt('predictions_train.csv', sess2.run(prediction_save, feed_dict={self.net.is_training:1}), delimiter=',')
#                                                 np.savetxt('net_current_V_train.csv', sess2.run(self.net.current_V, feed_dict={self.net.is_training:1}), delimiter=',')
#                                                 np.savetxt('correct_train.csv', sess2.run(correct, feed_dict={self.net.is_training:1}), delimiter=',')
#                                                 return tf.constant(0)
#                                             def save_test():
#                                                 np.savetxt('predictions_test.csv', sess2.run(prediction_save, feed_dict={self.net.is_training:0}), delimiter=',')
#                                                 np.savetxt('net_current_V_test.csv', sess2.run(self.net.current_V, feed_dict={self.net.is_training:0}), delimiter=',')
#                                                 np.savetxt('correct_test.csv', sess2.run(correct, feed_dict={self.net.is_training:0}), delimiter=',')
#                                                 return tf.constant(0)
#                                             _ = tf.cond(self.net.is_training, save_train, save_test)
#                                             return tf.constant(0)
#                                         def pass_fn():
#                                             return tf.constant(0)

#                                         _ = tf.cond(tf.less(self.max_acc_test, accuracy), save, pass_fn)
#                                         sess2.close()
                                last_summary = time.time()            
                            if (i-1) % 100 == 0:
                                total_training = 0.0
                                total_testing = 0.0
                                start_at = time.time()
                        if i % self.iterations_per_test == 0:
                            summary = sess.run(summary_merged, feed_dict={self.net.is_training:0})
                            if self.debug == False:
                                test_writer.add_summary(summary, i)
                            self.print_ext('Test Step %d Finished' % i)
                    except KeyboardInterrupt as err:
                        self.print_ext('Training interrupted at %d' % i)
                        wasKeyboardInterrupt = True
                        raisedEx = err
                    finally:
                        if i > 0 and self.debug == False:
                            self.save_model(sess, saver, i)
                        self.print_ext('Training completed, starting cleanup!')
                        coord.request_stop()
                        coord.join(threads)
                        self.print_ext('Cleanup completed!')
                        if wasKeyboardInterrupt:
                            raise raisedEx

                    return sess.run([self.max_r2_test, self.net.global_step]) if is_regression else sess.run([self.max_acc_test, self.net.global_step])
            else:
                self.print_ext('Model "%s" already trained!' % self.model_name)
                return self.get_max_accuracy()