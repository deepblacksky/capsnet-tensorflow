import tensorflow as tf

flags = tf.app.flags

############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'data/mnist', 'the path for dataset')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 10, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('test_sum_freq', 400, 'the frequency of saving test summary(step)')
flags.DEFINE_integer('save_freq', 1, 'the frequency of saving model(epoch)')
flags.DEFINE_string('results', 'results', 'path for saving results')


############################
#   hyper parameter    #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('num_classes', 10, 'the number of classes')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')
flags.DEFINE_integer('epoch', 30, 'epoch')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.392,
                   'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')

FLAGS = tf.app.flags.FLAGS
