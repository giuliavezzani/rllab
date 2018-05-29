import tensorflow as tf

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.networks.nn import MLPFunctionMultiHead


class NNFunctionMultiHead(MLPFunctionMultiHead):
    def __init__(self,
                 env_spec,
                 hidden_layer_sizes=None,
                 hidden_layer_sizes_extra=None,
                 name='function',
                 task=0,
                 num_tasks=2):
        Serializable.quick_init(self, locals())

        self._Do = env_spec.observation_space.flat_dim
        self._task = task
        self._num_tasks = num_tasks
        name = name
        self.hidden_layer_sizes = hidden_layer_sizes

        self._observations_ph = tf.placeholder(
            tf.float32, shape=[None, self._Do], name='observations')

        super(NNFunctionMultiHead, self).__init__(
            self._observations_ph,
            name=name,
            hidden_layer_sizes=hidden_layer_sizes,
            hidden_layer_sizes_extra=hidden_layer_sizes_extra,
            task=task)

    def output_for(self, observations, task, reuse=tf.AUTO_REUSE):
        return super(NNFunctionMultiHead, self)._output_for(
            observations, task=task, reuse=reuse)

    def eval(self, observations):
        return super(NNFunctionMultiHead, self)._eval(observations)
