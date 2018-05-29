import tensorflow as tf

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.policies.gaussian_conv_policy import GaussianConvPolicy


class MLPFunctionCNNMultihead():
    def __init__(self,
                env_spec,
                num_tasks,
                conv_args,
                heads_args,
    ):



        self.perception = GaussianConvPolicy(
                    name = 'perception',
                    env_spec=env_spec,
                    conv_filters=conv_args.conv_filters,
                    conv_filter_sizes=conv_args.conv_filter_sizes,
                    conv_strides=conv_args.conv_strides,
                    conv_pads=conv_args.conv_pads,
        )

        self._heads = []
        for t in range(num_tasks):
            self._heads.append(MLP(name='head-'+str(t),
                              output_dim=self.perception.action_space.flatten(),
                              hidden_sizes=heads_args.hidden_sizes,
                              hidden_nonlinearity = heads_args.hidden_nonlinearity,
                              output_nonlinearity = heads_args.output_nonlinearity,
                              input_layer = heads_args.input_layer,
                              input_shape = heads_args.input_shape
                            ))

    def output_perception(self, observations):
        flat_obs = self.perception.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self.perception._f_dist([flat_obs])]
        return [mean, log_std]

    def output_head(self, observations, task):
        flat_obs = self.perception.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self.perception._f_dist([flat_obs])]

        self._heads[task].input_layer.input_var = [mean, log_std]
        output_task = self._heads[task]._output
        return output_task
