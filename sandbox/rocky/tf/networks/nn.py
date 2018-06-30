import tensorflow as tf

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized


def shared_net(
        *inputs,
        layer_sizes,
        activation_fn=tf.nn.relu,
        output_nonlinearity=None,
        task=0):

    def bias(n_units, task):
        return tf.get_variable(
            name='bias',
            shape=n_units,
            initializer=tf.zeros_initializer())

    def linear(x, n_units, task, postfix=None):
        input_size = x.shape[-1].value
        weight_name = 'weight' + '_' + str(postfix) if postfix else 'weight'
        weight = tf.get_variable(
            name=weight_name,
            shape=(input_size, n_units),
            initializer=tf.contrib.layers.xavier_initializer())

        return tf.tensordot(x, weight, axes=[-1, 0])

    out = 0
    for i, layer_size in enumerate(layer_sizes):
        with tf.variable_scope('layer_{i}'.format(i=i)):
            if i == 0:
                for j, input_tensor in enumerate(inputs):
                    out += linear(input_tensor, layer_size, task, j)
            else:
                out = linear(out, layer_size, task)

            out += bias(layer_size, task)

            if i < len(layer_sizes) - 1 and activation_fn:
                out = activation_fn(out)

    if output_nonlinearity:
       out = output_nonlinearity(out)

    if len(layer_sizes) == 1:
        out = inputs

    return out

def head_net(
        *inputs,
        layer_sizes,
        layer_sizes_extra,
        activation_fn=tf.nn.relu,
        output_nonlinearity_extra=None,
        task=0):

    def bias(n_units, task):
        return tf.get_variable(
            name='bias',
            shape=n_units,
            initializer=tf.zeros_initializer())

    def linear(x, n_units, task, postfix=None):
        input_size = x.shape[-1].value
        weight_name = 'weight' + '_' + str(postfix) if postfix else 'weight'
        weight = tf.get_variable(
            name=weight_name,
            shape=(input_size, n_units),
            initializer=tf.contrib.layers.xavier_initializer())

        return tf.tensordot(x, weight, axes=[-1, 0])

    out = 0
    for i, layer_size in enumerate(layer_sizes_extra):
        with tf.variable_scope('layer_'+str(i+len(layer_sizes)).format(i=i)):
            if i == 0:
                for j, input_tensor in enumerate(inputs):
                    out += linear(input_tensor, layer_size, task, j)
            else:
                out = linear(out, layer_size, task)

            out += bias(layer_size, task)

            if i < len(layer_sizes_extra) - 1 and activation_fn:
                out = activation_fn(out)

    if output_nonlinearity_extra:
        out = output_nonlinearity(out)

    if len(layer_sizes_extra) == 1:
        out = inputs

    return out

class MLPFunctionMultiHead(Parameterized, Serializable):
    def __init__(self, *inputs, name, hidden_layer_sizes, hidden_layer_sizes_extra, task):
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        self._name = name
        self._inputs = inputs
        if not hidden_layer_sizes==None:
            self._layer_sizes = list(hidden_layer_sizes)
        else:
            self._layer_sizes = [1]
        if len(hidden_layer_sizes) == 1:
            self._layer_sizes = list(hidden_layer_sizes) + [1]
        if not hidden_layer_sizes_extra==None:
            self._layer_sizes_extra = list(hidden_layer_sizes_extra) + [1]
        else:
            self._layer_sizes_extra = [1]
        self._task = task

        self._output = []

        for task in range(self._num_tasks):
            self._output.append(self._output_for(*self._inputs, task=task))


    def _output_for(self, *inputs, task, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self._name, reuse=reuse):
            out = shared_net(
                *inputs,
                output_nonlinearity=None,
                layer_sizes=self._layer_sizes,
                task=task,
            )

        original_name = self._name
        self._name = self._name + '-' + str(task)
        with tf.variable_scope(self._name, reuse=reuse):
            out = head_net(
                out,
                output_nonlinearity_extra=None,
                layer_sizes=self._layer_sizes,
                layer_sizes_extra=self._layer_sizes_extra,
                task=task,
            )

        self._name = original_name
        return out[..., 0]

    def _output_for_shared(self, *inputs, task, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self._name, reuse=reuse):
            out = shared_net(
                *inputs,
                output_nonlinearity=None,
                layer_sizes=self._layer_sizes,
                task=task,
                #activation_fn=tf.nn.selu,
            )

        return out

    def _eval(self, *inputs, task):
        feeds = {pl: val for pl, val in zip(self._inputs, inputs)}

        return tf_utils.get_default_session().run(self._output[task], feeds)

    def get_params_internal(self, scope='', **tags):
        if len(tags) > 0:
            raise NotImplementedError

        scope += '/' + self._name if scope else self._name

        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
