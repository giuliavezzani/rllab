import tensorflow as tf
import re
import numpy as np

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """

  TOWER_NAME = 'tower'
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

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
    dtype =  tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

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
  dtype =  tf.float32

  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def cnn_shared(*inputs, batch_size,dropout, layer):

    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=None)

        conv = tf.nn.conv2d(inputs[0], kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    if layer == 1:
        return conv1

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    if layer == 2:
        return conv2
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    ### Let's see if dropout helps for bottleneck
    if dropout:
        pool2 = tf.nn.dropout(pool2, 0.8)

    merged = tf.summary.merge_all()

    return pool2, merged

def cnn_head(pool2, layer):
  with tf.variable_scope('local3') as scope:
      # Move everything into depth so we can perform a single matrix multiply.
      shape = pool2.get_shape().as_list()
      d = np.prod(shape[1:])
      reshape = tf.reshape(pool2, [-1, d])
      dim = reshape.get_shape()[1].value
      weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                            stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
      local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
      _activation_summary(local3)

  if layer == 3:
      return local3

# local4
  with tf.variable_scope('local4') as scope:
      weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                            stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
      _activation_summary(local4)

  if layer == 4:
      return local4

  with tf.variable_scope('output') as scope:
      weights = _variable_with_weight_decay('weights', [192, 1],
                                            stddev=1/192.0, wd=None)
      biases = _variable_on_cpu('biases', [1],
                              tf.constant_initializer(0.0))
      out = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
      _activation_summary(out)

  merged = tf.summary.merge_all()

  return out, merged


class MLPFunctionCNNMultihead(Parameterized, Serializable):
    def __init__(self, *inputs,  batch_size=128, dropout=False, name='function'):

        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        self._name = 'function'
        self._inputs = inputs
        self._batch_size = batch_size
        self._dropout = dropout

        for task in range(self._num_tasks):
            self._output, self._summary = self._output_for(*self._inputs, task=task)

    def _output_for(self, *inputs, task, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self._name, reuse=reuse):
            out, summary1 = cnn_shared(
                *inputs,
                batch_size=self._batch_size,
                layer=None,
                dropout=self._dropout
            )

        original_name = self._name
        self._name = self._name + '-' + str(task)
        with tf.variable_scope(self._name, reuse=reuse):
            out, summary2 = cnn_head(
                *inputs,
                layer=None,
            )

        self._name = original_name
        return out[..., 0], [summary1, summary2]

    def _output_for_shared(self, *inputs, task, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self._name, reuse=reuse):
            out, summary1 = cnn_shared(
                *inputs,
                batch_size=self._batch_size,
                layer=None,
                dropout=self._dropout
            )

        return out[..., 0], summary1

    def get_activation(self, *inputs, layer=None):
        with tf.variable_scope(self._name, reuse=True):
            return cnn_shared(*inputs,
                    batch_size=self._batch_size, layer=layer)

    def _eval(self, *inputs):
        feeds = {pl: val for pl, val in zip(self._inputs, inputs)}

        return tf_utils.get_default_session().run(self._output, feeds)

    def get_params_internal(self, scope='', **tags):
        if len(tags) > 0:
            raise NotImplementedError

        scope += '/' + self._name if scope else self._name

        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
