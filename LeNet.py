import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

class LeNetConfig(object):
    def __init__(self, input_dim, num_classes, params):
        self.W, self.H, self.C = input_dim
        self.num_classes = num_classes

        self.conv1_num_filters = params['conv1_num_filters'] if 'conv1_num_filters' in params else 6
        self.conv1_filter_size = params['conv1_filter_size'] if 'conv1_filter_size' in params else 5
        self.conv1_stride = params['conv1_stride'] if 'conv1_stride' in params else 1
        self.pool1_size = params['pool1_size'] if 'pool1_size' in params else 2
        self.pool1_stride = params['pool1_stride'] if 'pool1_stride' in params else 2

        self.conv2_num_filters = params['conv2_num_filters'] if 'conv2_num_filters' in params else 16
        self.conv2_filter_size = params['conv2_filter_size'] if 'conv2_filter_size' in params else 5
        self.conv2_stride = params['conv2_stride'] if 'conv2_stride' in params else 1
        self.pool2_size = params['pool2_size'] if 'pool2_size' in params else 2
        self.pool2_stride = params['pool2_stride'] if 'pool2_stride' in params else 2

        self.fc1_depth = params['fc1_depth'] if 'fc1_depth' in params else 120
        self.fc2_depth = params['fc2_depth'] if 'fc2_depth' in params else 84

        self.use_bn = params['use_bn'] if 'use_bn' in params else False
        self.dropout_prob = params['dropout_prob'] if 'dropout_prob' in params else 1.0
        self.l2 = params['l2'] if 'l2' in params else 0.0

        self.mu = params['mu'] if 'mu' in params else 0
        self.sigma = params['sigma'] if 'sigma' in params else 0.1

        self.batch_size = params['batch_size'] if 'batch_size' in params else 128
        self.num_epochs = params['num_epochs'] if 'num_epochs' in params else 5

        self.lr_start = params['lr_start'] if 'lr_start' in params else 1e-3
        self.lr_decay_steps = params['lr_decay_steps'] if 'lr_decay_steps' in params else 100000
        self.lr_decay_rate = params['lr_decay_rate'] if 'lr_decay_rate' in params else 0.96


class LeNetParameters(object):
    def __init__(self, net_config, init_weights=True):
        self.net_config = net_config

        if init_weights:
            mu = net_config.mu
            sigma = net_config.sigma

            conv1_out_w, conv1_out_h = self.calc_conv_out_size(net_config.W, net_config.H, net_config.conv1_filter_size, net_config.conv1_stride)
            pool1_out_w, pool1_out_h = self.calc_pool_out_size(conv1_out_w, conv1_out_h, net_config.pool1_size, net_config.pool1_size, net_config.pool2_stride)

            conv2_out_w, conv2_out_h = self.calc_conv_out_size(pool1_out_w, pool1_out_h, net_config.conv2_filter_size, net_config.conv2_stride)
            pool2_out_w, pool2_out_h = self.calc_pool_out_size(conv2_out_w, conv2_out_h, net_config.pool2_size, net_config.pool2_size, net_config.pool2_stride)

            weights = 5 * [None]
            biases = 5 * [None]

            # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
            weights[0] = self.weight_variable(shape=(net_config.conv1_filter_size, net_config.conv1_filter_size, net_config.C, net_config.conv1_num_filters), mean=mu, stddev=sigma)
            biases[0] = self.bias_variable((net_config.conv1_num_filters))

            # Layer 2: Convolutional. Output = 10x10x16.
            weights[1] = self.weight_variable(shape=(net_config.conv2_filter_size, net_config.conv2_filter_size, net_config.conv1_num_filters, net_config.conv2_num_filters), mean=mu, stddev=sigma)
            biases[1] = self.bias_variable((net_config.conv2_num_filters))

            # Layer 3: Fully Connected. Input = 400. Output = 120.
            weights[2] = self.weight_variable(shape=(pool2_out_w * pool2_out_h * net_config.conv2_num_filters, net_config.fc1_depth), mean=mu, stddev=sigma)
            biases[2] = self.bias_variable((net_config.fc1_depth))

            # Layer 4: Fully Connected. Input = 120. Output = 84.
            weights[3] = self.weight_variable(shape=(net_config.fc1_depth, net_config.fc2_depth), mean=mu, stddev=sigma)
            biases[3] = self.bias_variable((net_config.fc2_depth))

            # Layer 5: Fully Connected. Input = 84. Output = n_classes.
            weights[4] = self.weight_variable(shape=(net_config.fc2_depth, net_config.num_classes), mean=mu, stddev=sigma)#tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
            biases[4] = self.bias_variable((net_config.num_classes))#, default_value=0)#tf.Variable(tf.zeros(n_classes))

            self.weights = weights
            self.biases = biases
        else:
            self.weights = None
            self.biases = None

    def calc_conv_out_size(self, input_w, input_h, filter_size, conv_stride):
        pad = 0#(filter_size - 1) / 2
        out_w = 1 + (input_w + 2 * pad - filter_size) / conv_stride
        out_h = 1 + (input_h + 2 * pad - filter_size) / conv_stride
        return (int(out_w), int(out_h))

    def calc_pool_out_size(self, input_w, input_h, pool_w, pool_h, pool_stride):
        out_w = (input_w - pool_w) / pool_stride + 1
        out_h = (input_h - pool_h) / pool_stride + 1
        return (int(out_w), int(out_h))


    def weight_variable(self, shape, mean=0, stddev=0.1):
        # return tf.Variable(tf.truncated_normal(shape, stddev=stddev))
        return np.random.normal(mean, stddev, shape).astype(np.float32)

    def bias_variable(self, shape, default_value=0.1):
        # return tf.Variable(tf.constant(default_value, shape=shape))
        return np.full(shape, default_value).astype(np.float32)

class LeNet(object):
    def __init__(self, model):
        self.net_config = model.net_config

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
            self.x = tf.placeholder(tf.float32, shape=(None, model.net_config.W, model.net_config.H, model.net_config.C))
            self.y = tf.placeholder(tf.int32, shape=(None))
            self.one_hot_y = tf.one_hot(self.y, model.net_config.num_classes)

            # Using placeholder allows us to use Dropout only during Training process (not Evaluating)
            self.dropout = tf.placeholder(tf.float32)

            self.is_training_mode = tf.placeholder(tf.bool)

            self.lr_start = tf.placeholder(tf.float32)
            self.lr_decay_steps = tf.placeholder(tf.float32)
            self.lr_decay_rate = tf.placeholder(tf.float32)

            # Build Tensors for Weights and biases
            self.weights = [tf.Variable(w) for w in model.weights]
            self.biases = [tf.Variable(b) for b in model.biases]

            (self.loss, self.optimizer, self.accuracy) = self.build_LeNet()

    def build_LeNet(self):
        # Layer 1
        conv1 = tf.nn.conv2d(self.x, self.weights[0], strides=[1, self.net_config.conv1_stride, self.net_config.conv1_stride, 1], padding='VALID') + self.biases[0]
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, self.net_config.pool1_size, self.net_config.pool1_size, 1], strides=[1, self.net_config.pool1_stride, self.net_config.pool1_stride, 1], padding='VALID')

        # Layer 2
        conv2 = tf.nn.conv2d(conv1, self.weights[1], strides=[1, self.net_config.conv2_stride, self.net_config.conv2_stride, 1], padding='VALID') + self.biases[1]
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, self.net_config.pool2_size, self.net_config.pool2_size, 1], strides=[1, self.net_config.pool2_stride, self.net_config.pool2_stride, 1], padding='VALID')

        # Flatten
        fc0 = flatten(conv2)

        # Layer 3
        fc1 = tf.matmul(fc0, self.weights[2]) + self.biases[2]
        if self.net_config.use_bn:
            fc1 = tf.contrib.layers.batch_norm(fc1, center=True, scale=True, is_training=self.is_training_mode)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, self.dropout)


        # Layer 4
        fc2 = tf.matmul(fc1, self.weights[3]) + self.biases[3]
        if self.net_config.use_bn:
            fc2 = tf.contrib.layers.batch_norm(fc2, center=True, scale=True, is_training=self.is_training_mode)
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, self.dropout)

        # Layer 5 - Readout Layer
        logits = tf.matmul(fc2, self.weights[4]) + self.biases[4]

        # Loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, self.one_hot_y)
        loss = tf.reduce_mean(cross_entropy)

        # L2 Regularization
        if self.net_config.l2 > 0:
            for w in self.weights:
                loss += self.net_config.l2 * tf.nn.l2_loss(w)

        # Accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Optimizer
        # tf.contrib.layers.batch_norm doc says:
        # When is_training is True the moving_mean and moving_variance need to be
        # updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
        # they need to be added as a dependency to the `train_op`
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr_start, global_step, self.lr_decay_steps, self.lr_decay_rate, staircase=True)
            # Passing global_step to minimize() will increment it at each step.
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        return (loss, optimizer, accuracy_operation)



    def update_from_model(self, model):
        with self.graph.as_default():
            self.weights = [tf.Variable(w) for w in model.weights]
            self.biases = [tf.Variable(b) for b in model.biases]

            # rebuild layers
            (self.loss, self.optimizer, self.accuracy) = self.build_LeNet()

    def update_to_model(self, model):
        model.weights = [w.eval() for w in self.weights]
        model.biases = [b.eval() for b in self.biases]

    def evaluate(self, session, X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        for offset in range(0, num_examples, self.net_config.batch_size):
            batch_x, batch_y = X_data[offset:offset + self.net_config.batch_size], y_data[offset:offset + self.net_config.batch_size]
            accuracy = session.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: 1.0, self.is_training_mode: False})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def predict(self, X_test, y_test):
        with tf.Session(graph=self.graph) as session:
            session.run(tf.global_variables_initializer())
            test_accuracy = self.evaluate(session, X_test, y_test)
            return test_accuracy



class LeNetSolver(object):
    def __init__(self, leNet, train_dataset, train_labels, valid_dataset, valid_labels, debug=True):
        self.leNet = leNet
        self.net_config = leNet.net_config
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.valid_dataset = valid_dataset
        self.valid_labels = valid_labels

        self.debug = debug


    def train(self):
        best_valid_loss = 0
        best_valid_accuracy = 0
        best_valid_params = LeNetParameters(self.leNet.net_config, init_weights=False)

        history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}

        with tf.Session(graph=self.leNet.graph) as session:
            session.run(tf.global_variables_initializer())
            num_examples = len(self.train_dataset)

            print("Training...")
            print()
            for i in range(self.net_config.num_epochs):
                X_train, y_train = shuffle(self.train_dataset, self.train_labels)
                train_total_loss = 0.0
                for offset in range(0, num_examples, self.net_config.batch_size):
                    end = offset + self.net_config.batch_size
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    feed_dict = {self.leNet.x: batch_x, self.leNet.y: batch_y,
                                 self.leNet.dropout: self.net_config.dropout_prob,
                                 self.leNet.is_training_mode: True,
                                 self.leNet.lr_start: self.net_config.lr_start, self.leNet.lr_decay_steps: self.net_config.lr_decay_steps,
                                 self.leNet.lr_decay_rate: self.net_config.lr_decay_rate}

                    _, loss_val = session.run([self.leNet.optimizer, self.leNet.loss], feed_dict=feed_dict)
                    train_total_loss += (loss_val * len(batch_x) )

                train_total_loss /= num_examples

                train_accuracy = self.leNet.evaluate(session, self.train_dataset, self.train_labels)
                valid_accuracy = self.leNet.evaluate(session, self.valid_dataset, self.valid_labels)

                valid_loss = self.leNet.loss.eval(feed_dict={self.leNet.x: self.valid_dataset,
                                                             self.leNet.y: self.valid_labels,
                                                             self.leNet.dropout: 1.0, self.leNet.is_training_mode: False})

                if self.debug:
                    print("EPOCH {} ...".format(i + 1))
                    #print("Minibatch Loss: %f" % loss_val)
                    print("Train Loss: %f" % train_total_loss)
                    print("Train Accuracy = {:.3f}".format(train_accuracy))
                    print("Validation Loss = {:.3f}".format(valid_loss))
                    print("Validation Accuracy = {:.3f}".format(valid_accuracy))
                    print()

                # Keep track of the best model
                if valid_accuracy > best_valid_accuracy:
                    best_valid_loss = valid_loss
                    best_valid_accuracy = valid_accuracy
                    self.leNet.update_to_model(best_valid_params)

                # store history
                history['train_loss'].append(train_total_loss)
                history['train_acc'].append(train_accuracy)
                history['valid_loss'].append(valid_loss)
                history['valid_acc'].append(valid_accuracy)

            #if self.debug:
            print("Best Valid Accuracy: {:.1f}% \n".format(best_valid_accuracy * 100))

            return (history, best_valid_loss, best_valid_accuracy, best_valid_params)


