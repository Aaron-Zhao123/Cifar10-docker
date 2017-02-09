import cifar10
import tensorflow as tf
import numpy as np
import sys
import os
import pickle
import time

from cifar10 import img_size, num_channels, num_classes
def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    Returns:
    Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def initialize_variables(exist, file_name):
    NUM_CHANNELS = 3
    IMAGE_SIZE = 32
    NUM_CLASSES = 10
    keys = ['cov1','cov2','fc1','fc2','fc3']
    if (exist == 1):
        with open(file_name, 'rb') as f:
            (weights_val, biases_val) = pickle.load(f)
        weights = {
            'cov1': tf.Variable(weights_val['cov1']),
            'cov2': tf.Variable(weights_val['cov2']),
            'fc1': tf.Variable(weights_val['fc1']),
            'fc2': tf.Variable(weights_val['fc2']),
            'fc3': tf.Variable(weights_val['fc3'])
        }
        biases = {
            'cov1': tf.Variable(biases_val['cov1']),
            'cov2': tf.Variable(biases_val['cov2']),
            'fc1': tf.Variable(biases_val['fc1']),
            'fc2': tf.Variable(biases_val['fc2']),
            'fc3': tf.Variable(biases_val['fc3'])
        }
    else:
        print('hello?')
        weights = {
            # 'cov1': _variable_with_weight_decay('weights_cov1' ,
            #                                     shape = [5, 5, NUM_CHANNELS, 64],
            #                                     stddev = 5e-2,
            #                                     wd=0.004),
            # 'cov2': _variable_with_weight_decay('weights_cov2' ,
            #                                     shape = [5, 5, 64, 64],
            #                                     stddev = 5e-2,
            #                                     wd=0.004),
            # 'fc1': _variable_with_weight_decay('weights_fc1' ,
            #                                     shape = [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 384],
            #                                     stddev = 5e-2,
            #                                     wd=0.004),
            # 'fc2': _variable_with_weight_decay('weights_fc2' ,
            #                                     shape = [384, 192],
            #                                     stddev = 5e-2,
            #                                     wd=0.004),
            # 'fc3': _variable_with_weight_decay('weights_fc3' ,
            #                                     shape = [192, NUM_CLASSES],
            #                                     stddev = 1/192.0,
            #                                     wd=0.0)
            'cov1': tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 64],
                                                        stddev=5e-2)),
            'cov2': tf.Variable(tf.truncated_normal([5, 5, 64, 64],
                                                        stddev=5e-2)),
            'fc1': tf.Variable(tf.truncated_normal([6 * 6 * 64, 384],
                                                        stddev=0.04)),
            'fc2': tf.Variable(tf.random_normal([384, 192],
                                                        stddev=0.04)),
            'fc3': tf.Variable(tf.random_normal([192, NUM_CLASSES],
                                                        stddev=1/192.0))
        }
        biases = {
            'cov1': tf.Variable(tf.constant(0.1, shape=[64])),
            'cov2': tf.Variable(tf.constant(0.1, shape=[64])),
            'cov2': tf.Variable(tf.constant(0.1, shape=[64])),
            'fc1': tf.Variable(tf.constant(0.1, shape=[384])),
            'fc2': tf.Variable(tf.constant(0.1, shape=[192])),
            'fc3': tf.Variable(tf.constant(0.0, shape=[NUM_CLASSES]))
        }
    return (weights, biases)

def cov_network(images, weights, biases, keep_prob):
    BATCH_SIZE = 128
    NUM_CLASSES = 10
    IMAGE_SIZE = 32
    # conv1
    conv = tf.nn.conv2d(images, weights['cov1'], [1, 1, 1, 1], padding='SAME')

    pre_activation = tf.nn.bias_add(conv, biases['cov1'])
    conv1 = tf.nn.relu(pre_activation)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

    # conv2
    conv = tf.nn.conv2d(norm1, weights['cov2'], [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases['cov2'])
    conv2 = tf.nn.relu(pre_activation)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    # Move everything into depth so we can perform a single matrix multiply.
    # dim = 1
    # for d in pool2.get_shape()[1:].as_list():
    #   dim *= d
    # print(pool2.get_shape().as_list())
    # reshape = tf.reshape(pool2, [-1, IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64])
    reshape = tf.reshape(pool2, [-1, 6 * 6 * 64])
    # reshape = tf.reshape(pool2, [BATCH_SIZE, dim])
    # print(reshape)

    local3 = tf.nn.relu(tf.matmul(reshape, weights['fc1']) + biases['fc1'])
    # dropout
    local3_drop = tf.nn.dropout(local3, keep_prob)
# # local4
    local4 = tf.nn.relu(tf.matmul(local3_drop, weights['fc2']) + biases['fc2'])
    local4_drop = tf.nn.dropout(local4, keep_prob)
# # We don't apply softmax here because
# # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
# # and performs the softmax internally for efficiency.
    softmax_linear = tf.add(tf.matmul(local4_drop, weights['fc3']), biases['fc3'])
    return softmax_linear

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

class training_data():
    def __init__ (self, images, labels):
        self.images = images
        self.labels = labels
        self.batch_cnt = 0
        self.data_size = len(images)
    def feed_next_batch(self, batch_size):
        start_pt = self.batch_cnt
        self.batch_cnt += batch_size
        if (self.batch_cnt <= self.data_size):
            return (self.images[start_pt:self.batch_cnt],
                    self.labels[start_pt:self.batch_cnt])
        else:
            self.batch_cnt = 0
            return (self.images[start_pt:self.data_size],
                    self.labels[start_pt:self.data_size])


def save_pkl_model(weights, biases, file_name):
    data_path = '/root/data/' + file_name
    name = '/root/data/' + file_name
    # name = os.path.join(data_path, "cifar-10-batches-py/", filename)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    keys = ['cov1','cov2','fc1','fc2','fc3']
    weights_val = {}
    biases_val = {}
    for key in keys:
        weights_val[key] = weights[key].eval()
        biases_val[key] = biases[key].eval()
    with open('/root/data/test.pkl', 'wb') as f:
        print('Created a pickle file')
        pickle.dump((weights_val, biases_val), f)

def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    num_channels = 3
    img_size_cropped = 24

    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image


def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)
    print(images)

    return images


def main():
    NUM_CLASSES = 10
    dropout = 0.5
    BATCH_SIZE = 128
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
    INITIAL_LEARNING_RATE = 0.1
    LEARNING_RATE_DECAY_FACTOR = 0.1
    NUM_EPOCHS_PER_DECAY = 350.0
    MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
    DISPLAY_FREQ = 20
    TRAIN_OR_TEST = 1
    # model_name = 'tmp_20160130.pkl'
    # model_name = 'data_sync/test20170203.pkl'
    #model_name = '20170205.pkl'
    model_name = './data/20170209.pkl'
    # model_name = 'data_sync/20170206.pkl'
    # model_name = 'test.pkl'
    # model_name = '../tf_official_docker/tmp.pkl'
    PREV_MODEL_EXIST = 1


    # cls_train returns as an integer, labels is the array
    cifar10.maybe_download_and_extract()
    class_names = cifar10.load_class_names()
    images_train, cls_train, labels_train = cifar10.load_training_data()
    t_data = training_data(images_train, labels_train)

    DATA_CNT = len(images_train)
    NUMBER_OF_BATCH = DATA_CNT / BATCH_SIZE

    training_data_list = []

    weights, biases = initialize_variables(PREV_MODEL_EXIST, model_name)
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    keep_prob = tf.placeholder(tf.float32)
    images = pre_process(x, TRAIN_OR_TEST)
    pred = cov_network(images, weights, biases, keep_prob)
    # print(pred)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred, y)
    loss_value = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()


    # global_step = tf.contrib.framework.get_or_create_global_step()

    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    # lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
    #                               global_step,
    #                               decay_steps,
    #                               LEARNING_RATE_DECAY_FACTOR,
    #                               staircase=True)
    #
    # opt = tf.train.GradientDescentOptimizer(lr)
    # grads = opt.compute_gradients(loss_value)
    #
  # Apply gradients.
    # train_step = opt.apply_gradients(grads, global_step=global_step)
    train_step = tf.train.GradientDescentOptimizer(INITIAL_LEARNING_RATE).minimize(loss_value)
    # variable_averages = tf.train.ExponentialMovingAverage(
    #   MOVING_AVERAGE_DECAY, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())


    init = tf.global_variables_initializer()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # restore model if exists
        # if (os.path.isfile("tmp_20160130/model.meta")):
        #     op = tf.train.import_meta_graph("tmp_20160130/model.meta")
        #     op.restore(sess,tf.train.latest_checkpoint('tmp_20160130/'))
        #     print ("model found and restored")
        start = time.time()
        if TRAIN_OR_TEST == 1:
            for i in range(0,100000):
                (batch_x, batch_y) = t_data.feed_next_batch(BATCH_SIZE)
                train_acc, cross_en = sess.run([accuracy, loss_value], feed_dict = {
                                x: batch_x,
                                y: batch_y,
                                keep_prob: 1.0})
                if (i % DISPLAY_FREQ == 0):
                    print('This is the {}th iteration, time is {}'.format(
                        i,
                        time.time() - start
                    ))
                    print("accuracy is {} and cross entropy is {}".format(
                        train_acc,
                        cross_en
                    ))
                    if (i%(DISPLAY_FREQ*50) == 0 and i != 0 ):
                        save_pkl_model(weights, biases, model_name)
                        # saver.save(sess, "tmp_20160130/model")
                        print("saved the network")
                _ = sess.run(train_step, feed_dict = {
                                x: batch_x,
                                y: batch_y,
                                keep_prob: dropout})
        images_test, cls_test, labels_test = cifar10.load_test_data()
        test_acc = sess.run(accuracy, feed_dict = {
                                x: images_test,
                                y: labels_test,
                                keep_prob: 1.0})
        # save_pkl_model(weights, biases, model_name)
        print("test accuracy is {}".format(test_acc))


if __name__ == '__main__':
    main()
