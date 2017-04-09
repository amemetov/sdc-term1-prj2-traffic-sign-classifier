import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

class LeNetConfig(object):
    def __init__(self, input_dim, num_classes, params):
        self.W, self.H, self.C = input_dim
        self.num_classes = num_classes

        if 'conv' in params:
            self.conv = params['conv']
        else:
            self.conv = [{'num_filters': 6, 'filter_size': 5, 'stride': 1, 'pool_size': 2, 'pool_stride': 2},
                         {'num_filters': 16, 'filter_size': 5, 'stride': 1, 'pool_size': 2, 'pool_stride': 2}]

        # validate conv layers
        for cl in self.conv:
            if 'num_filters' not in cl:
                cl['num_filters'] = 6
            if 'filter_size' not in cl:
                cl['filter_size'] = 5
            if 'stride' not in cl:
                cl['stride'] = 1
            if 'pool_size' not in cl:
                cl['pool_size'] = 2
            if 'pool_stride' not in cl:
                cl['pool_stride'] = 2

        if 'fc' in params:
            self.fc = params['fc']
        else:
            self.fc = [120, 84]

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

        self.optimizer = params['optimizer'] if 'optimizer' in params else 'adam'


class LeNetWeights(object):
    def __init__(self, net_config, init_weights=True):
        self.net_config = net_config

        self.weights = None
        self.biases = None

        if init_weights:
            mu = net_config.mu
            sigma = net_config.sigma

            weights = []
            biases = []

            conv_num = len(net_config.conv)
            fc_num = len(net_config.fc)

            # Convolutional Layers Weights and compute Pool Out Size
            conv = net_config.conv
            pool_out_w, pool_out_h = net_config.W, net_config.H
            conv_prev_num_filters = net_config.C
            for i in range(conv_num):
                weights.append(self.weight_variable(shape=(conv[i]['filter_size'], conv[i]['filter_size'], conv_prev_num_filters, conv[i]['num_filters']), mean=mu, stddev=sigma))
                biases.append(self.bias_variable((conv[i]['num_filters']), 0))

                conv_prev_num_filters = conv[i]['num_filters']
                conv_out_w, conv_out_h = self.calc_conv_out_size(pool_out_w, pool_out_h, conv[i]['filter_size'], conv[i]['stride'])
                if conv[i]['pool_size'] > 0:
                    pool_out_w, pool_out_h = self.calc_pool_out_size(conv_out_w, conv_out_h, conv[i]['pool_size'], conv[i]['pool_size'], conv[i]['pool_stride'])
                else:
                    pool_out_w, pool_out_h = conv_out_w, conv_out_h

                    # FullyConnected Layers Weights
            fc = net_config.fc
            fc_in_size = pool_out_w * pool_out_h * conv[-1]['num_filters']
            for i in range(fc_num):
                fc_out_size = fc[i]
                weights.append(self.weight_variable(shape=(fc_in_size, fc_out_size), mean=mu, stddev=sigma))
                biases.append(self.bias_variable((fc_out_size), 0))
                fc_in_size = fc_out_size

            # Readout Layer
            weights.append(self.weight_variable(shape=(fc_in_size, net_config.num_classes), mean=mu, stddev=sigma))
            biases.append(self.bias_variable((net_config.num_classes)))

            self.weights = weights
            self.biases = biases

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
        return np.random.normal(mean, stddev, shape).astype(np.float32)

    def bias_variable(self, shape, default_value=0.1):
        return np.full(shape, default_value).astype(np.float32)


class LeNet(object):
    def __init__(self, net_config):
        self.net_config = net_config

        self.graph = None
        self.session = None

        self._build_graph(net_config, LeNetWeights(net_config))

    def _build_graph(self, net_config, net_weights):
        self.close_session()

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
            self.x = tf.placeholder(tf.float32, shape=(None, net_config.W, net_config.H, net_config.C))
            self.y = tf.placeholder(tf.int32, shape=(None))
            self.one_hot_y = tf.one_hot(self.y, net_config.num_classes)

            self.is_training_mode = tf.placeholder(tf.bool)

            # Using placeholder allows us to use Dropout only during Training process (not Evaluating)
            self.dropout = tf.placeholder(tf.float32)

            self.lr_start = tf.placeholder(tf.float32)
            self.lr_decay_steps = tf.placeholder(tf.float32)
            self.lr_decay_rate = tf.placeholder(tf.float32)

            # Build Tensors for Weights and biases
            self.weights = [tf.Variable(w, name='w_' + str(i)) for w, i in zip(net_weights.weights, range(len(net_weights.weights)))]
            self.biases = [tf.Variable(b, name='b_' + str(i)) for b, i in zip(net_weights.biases, range(len(net_weights.biases)))]

            self._build_layers()

            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()

        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init_op)#tf.global_variables_initializer())

    def _build_layers(self):
        conv_num = len(self.net_config.conv)
        fc_num = len(self.net_config.fc)

        # Convolutional Layers
        conv = self.net_config.conv
        conv_input = self.x
        for i in range(conv_num):
            tf_conv = tf.add(tf.nn.conv2d(conv_input, self.weights[i],
                                          strides=[1, conv[i]['stride'], conv[i]['stride'], 1],
                                          padding='VALID'), self.biases[i], name='conv_{0}'.format(i+1))
            tf_relu = tf.nn.relu(tf_conv, name='conv_{0}_relu'.format(i+1))
            if conv[i]['pool_size'] > 0:
                tf_pool = tf.nn.max_pool(tf_relu, ksize=[1, conv[i]['pool_size'], conv[i]['pool_size'], 1],
                                         strides=[1, conv[i]['pool_stride'], conv[i]['pool_stride'], 1],
                                         padding='VALID', name='conv_{0}_pool'.format(i+1))
                conv_input = tf_pool
            else:
                conv_input = tf_relu

        # Flatten
        fc0 = flatten(conv_input)

        # FC Layers
        fc = self.net_config.fc
        fc_input = fc0
        for i in range(fc_num):
            w_i = i + conv_num
            tf_fc = tf.add(tf.matmul(fc_input, self.weights[w_i]), self.biases[w_i], name='fc_{0}'.format(i+1))
            if self.net_config.use_bn:
                tf_fc = tf.contrib.layers.batch_norm(tf_fc, center=True, scale=True, is_training=self.is_training_mode)
            tf_fc_relu = tf.nn.relu(tf_fc, name='fc_{0}_relu'.format(i+1))
            tf_fc_dropout = tf.nn.dropout(tf_fc_relu, self.dropout, name='fc_{0}_dropout'.format(i+1))
            fc_input = tf_fc_dropout

        # Readout Layer
        logits = tf.add(tf.matmul(fc_input, self.weights[-1]), self.biases[-1], name='logits')

        # Loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, self.one_hot_y, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')

        # L2 Regularization
        if self.net_config.l2 > 0:
            for w in self.weights:
                loss += self.net_config.l2 * tf.nn.l2_loss(w)

        # Prediction and Accuracy
        predict_op = tf.argmax(logits, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(self.one_hot_y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
            optimizer = self._create_optimizer(learning_rate).minimize(loss, global_step=global_step)

        # store necessary tensors for further using
        self.logits = logits
        self.predict = predict_op
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy_op

    def _create_optimizer(self, learning_rate):
        if self.net_config.optimizer == 'gd':
            return tf.train.GradientDescentOptimizer(learning_rate)
        if self.net_config.optimizer == 'adadelta':
            return tf.train.AdadeltaOptimizer(learning_rate)
        if self.net_config.optimizer == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate)
        if self.net_config.optimizer == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate, 0.9)
        elif self.net_config.optimizer == 'ftrl':
            return tf.train.FtrlOptimizer(learning_rate)
        elif self.net_config.optimizer == 'rmsp':
            return tf.train.RMSPropOptimizer(learning_rate)
        else:# self.net_config.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate)

    def close_session(self):
        if self.session is not None:
            self.session.close()

    def save(self, save_path):
        self.saver.save(sess=self.session, save_path=save_path)

    def restore(self, save_path):
        self.saver.restore(sess=self.session, save_path=save_path)

    def load_weights(self, weights):
        self._build_graph(self.net_config, weights)

    def get_weights(self):
        weights = LeNetWeights(self.net_config, init_weights=False)
        weights.weights = [w.eval(session=self.session) for w in self.weights]
        weights.biases = [b.eval(session=self.session) for b in self.biases]
        return weights

    def eval_loss(self, X, y):
        num_examples = len(X)
        total_loss = 0
        for offset in range(0, num_examples, self.net_config.batch_size):
            batch_x, batch_y = X[offset:offset + self.net_config.batch_size], y[offset:offset + self.net_config.batch_size]
            loss = self.session.run(self.loss, feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: 1.0, self.is_training_mode: False})
            total_loss += (loss * len(batch_x))
        return total_loss / num_examples

    def eval_accuracy(self, X, y):
        num_examples = len(X)
        total_accuracy = 0
        for offset in range(0, num_examples, self.net_config.batch_size):
            batch_x, batch_y = X[offset:offset + self.net_config.batch_size], y[offset:offset + self.net_config.batch_size]
            accuracy = self.session.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: 1.0, self.is_training_mode: False})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def predict_class(self, X):
        return self.session.run(self.predict, feed_dict={self.x: X, self.dropout: 1.0, self.is_training_mode: False})

    def predict_probabilities(self, X):
        return self.session.run(self.logits, feed_dict={self.x: X, self.dropout: 1.0, self.is_training_mode: False})

    def activation(self, image, var_name):
        tf_activation = self.graph.get_tensor_by_name(var_name + ":0")
        return tf_activation.eval(session=self.session, feed_dict={self.x: np.array(image), self.dropout: 1.0, self.is_training_mode: False})

    def fit(self, X_train, y_train, X_valid, y_valid, debug=True):
        steps_per_epoch = int(len(X_train) / self.net_config.batch_size)
        return LeNetSolver(self, TrainDataSource(X_train, y_train), steps_per_epoch, X_valid, y_valid, debug=debug).train()

    def fit_generator(self, train_generator, steps_per_epoch, X_valid, y_valid, debug=True):
        return LeNetSolver(self, TrainAugmentDataSource(train_generator), steps_per_epoch, X_valid, y_valid, debug=debug).train()


class TrainDataSource(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.offset = 0

    def start_epoch(self):
        self.X, self.y = shuffle(self.X, self.y)
        self.offset = 0

    def next_batch(self, batch_size):
        end = min(self.offset + batch_size, self.X.shape[0])
        X_batch, y_batch = self.X[self.offset:end], self.y[self.offset:end]
        self.offset += batch_size
        return X_batch, y_batch


class TrainAugmentDataSource(object):
    def __init__(self, train_generator):
        self.train_generator = train_generator

    def start_epoch(self):
        pass

    def next_batch(self, batch_size):
        return self.train_generator.next_batch(batch_size)


class LeNetSolver(object):
    def __init__(self, leNet, train_data_source, steps_per_epoch, valid_dataset, valid_labels, debug=True):
        self.leNet = leNet
        self.net_config = leNet.net_config
        self.train_data_source = train_data_source
        self.steps_per_epoch = steps_per_epoch
        self.valid_dataset = valid_dataset
        self.valid_labels = valid_labels

        self.debug = debug


    def train(self):
        best_valid_loss = 0
        best_valid_accuracy = 0
        best_valid_weights = None

        history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}

        num_examples = self.steps_per_epoch * self.net_config.batch_size
        train_dataset = np.ndarray([num_examples, self.net_config.W, self.net_config.H, self.net_config.C])
        train_labels = np.ndarray([num_examples])

        print("Training...")
        print()
        for i in range(self.net_config.num_epochs):
            train_total_loss = 0.0

            self.train_data_source.start_epoch()

            for s in range(self.steps_per_epoch):
                #print('Step: {0}'.format(s))
                batch_x, batch_y = self.train_data_source.next_batch(self.net_config.batch_size)

                start = s*self.net_config.batch_size
                end = start + self.net_config.batch_size
                train_dataset[start:end] = batch_x
                train_labels[start:end] = batch_y

                feed_dict = {self.leNet.x: batch_x, self.leNet.y: batch_y,
                             self.leNet.dropout: self.net_config.dropout_prob,
                             self.leNet.is_training_mode: True,
                             self.leNet.lr_start: self.net_config.lr_start,
                             self.leNet.lr_decay_steps: self.net_config.lr_decay_steps,
                             self.leNet.lr_decay_rate: self.net_config.lr_decay_rate}

                _, loss_val = self.leNet.session.run([self.leNet.optimizer, self.leNet.loss], feed_dict=feed_dict)
                train_total_loss += (loss_val * len(batch_x))

            train_total_loss /= num_examples

            train_accuracy = self.leNet.eval_accuracy(train_dataset, train_labels)
            valid_accuracy = self.leNet.eval_accuracy(self.valid_dataset, self.valid_labels)
            valid_loss = self.leNet.eval_loss(self.valid_dataset, self.valid_labels)

            if self.debug:
                print("EPOCH {} ...".format(i + 1))
                print("Train Loss: %f" % train_total_loss)
                print("Train Accuracy = {:.3f}".format(train_accuracy))
                print("Validation Loss = {:.3f}".format(valid_loss))
                print("Validation Accuracy = {:.3f}".format(valid_accuracy))
                print()

            # Keep track of the best model
            if valid_accuracy > best_valid_accuracy:
                best_valid_loss = valid_loss
                best_valid_accuracy = valid_accuracy
                best_valid_weights = self.leNet.get_weights()

            # store history
            history['train_loss'].append(train_total_loss)
            history['train_acc'].append(train_accuracy)
            history['valid_loss'].append(valid_loss)
            history['valid_acc'].append(valid_accuracy)


        # Update weights of leNet with best values
        self.leNet.load_weights(best_valid_weights)

        #if self.debug:
        print("Best Valid Accuracy: {:.1f}% \n".format(best_valid_accuracy * 100))

        return (history, best_valid_loss, best_valid_accuracy)
