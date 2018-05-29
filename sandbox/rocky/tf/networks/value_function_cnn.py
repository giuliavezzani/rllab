import tensorflow as tf

from rllab.core.serializable import Serializable
from networks.cnn import MLPFunctionCNNMultihead



class CNNFunctionMultihead(MLPFunctionCNNMultihead):
    def __init__(self,
                 dim_img,
                 dropout,
                 name='function',
                 num_tasks=2):
        Serializable.quick_init(self, locals())

        self._dim_img = dim_img
        name = name
        self._num_tasks = num_tasks
        self._dropout = dropout

        self._observations_ph = tf.placeholder(
            tf.float32, shape=[None, self._dim_img, self._dim_img, 3], name='observations')

        super(CNNFunctionMultihead, self).__init__(self._observations_ph, dropout=self._dropout, name=name)

    def _output_for(self, observations,task, reuse=tf.AUTO_REUSE):
        return super(CNNFunctionMultihead, self)._output_for(
            observations, task=task, reuse=reuse)

    def _output_for_shared(self, observations,task, reuse=tf.AUTO_REUSE):
        return super(CNNFunctionMultihead, self)._output_for_shared(
            observations, task=task, reuse=reuse)

    def eval(self, observations):
        return super(CNNFunctionMultihead, self)._eval(observations)
