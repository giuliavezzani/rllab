


from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.first_order_optimizer_multitask import FirstOrderOptimizerMultiTask
#from sandbox.rocky.tf.optimizers.lbfgs_optimizer import LbfgsOptimizer
#from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt_bc import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf


class BC(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            num_tasks=1,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = FirstOrderOptimizerMultiTask(max_epochs=1)
        #import IPython
        #IPython.embed()
        self.optimizer = optimizer

        self.optimizer_debug = FirstOrderOptimizerMultiTask(max_epochs=1)
        self.step_size = step_size
        self.num_tasks = num_tasks

        super(BC, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = []
        action_var = []

        for t in range(self.num_tasks):
            obs_var.append(self.env.observation_space.new_tensor_variable(
                'obs'+str(t),
                extra_dims=1 + is_recurrent,
            ))
            action_var.append(self.env.action_space.new_tensor_variable(
                'action'+str(t),
                extra_dims=1 + is_recurrent,
            ))
        #advantage_var = tensor_utils.new_tensor(
        #    'advantage',
        #    ndim=1 + is_recurrent,
        #    dtype=tf.float32,
        #)
        dist = self.policy.distribution
        old_dist_info_vars = []
        old_dist_info_vars_list = []
        state_info_vars = []
        state_info_vars_list = []
        dist_info_vars = []
        surr_loss = 0.0


        for t in range(self.num_tasks):
            old_dist_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
                for k, shape in dist.dist_info_specs
                })
            old_dist_info_vars_list.append([old_dist_info_vars[t][k] for k in dist.dist_info_keys])

            state_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
                for k, shape in self.policy.state_info_specs
                })
            state_info_vars_list.append([state_info_vars[t][k] for k in self.policy.state_info_keys])

            if is_recurrent:
                valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
            else:
                valid_var = None

            dist_info_vars.append(self.policy.dist_info_sym(tf.cast(obs_var[t], tf.float32), state_info_vars[t], t))

            # Behavior cloning loss
            print('Summing loss over task', t)
            #surr_loss += - tf.reduce_mean(dist.log_likelihood_sym(action_var[t], dist_info_vars[t]))

            surr_loss +=  tf.reduce_mean(tf.square(action_var[t] - dist_info_vars[t]['mean']))


        input_list = obs_var + action_var


        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            #leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )

        self.optimizer_debug.update_opt(
            loss=dist_info_vars[0]['log_std'],
            target=self.policy,
            #leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl2"
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data, batch_size):
        #all_input_values = tuple(ext.extract(
        #    samples_data,
        #    "observations", "actions", "advantages"
        #))

        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions"
        ))
        #agent_infos = samples_data["agent_infos"]
        #state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        #dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        #all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        #logger.log("Computing loss before")
        #loss_before = self.optimizer.loss(all_input_values)
        #logger.log("Computing KL before")
        #mean_kl_before = self.optimizer.constraint_val(all_input_values)
        logger.log("Optimizing")
        self.optimizer.optimize(all_input_values, batch_size=batch_size)
        #logger.log("Computing KL after")
        #mean_kl = self.optimizer.constraint_val(all_input_values)
        logger.log("Computing loss")
        loss = self.optimizer.loss(all_input_values)
        #logger.record_tabular('LossBefore', loss_before)
        #logger.record_tabular('LossAfter', loss_after)
        #logger.record_tabular('MeanKLBefore', mean_kl_before)
        #logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('Loss', loss)

        logger.log("Computing loss")
        loss_debug = self.optimizer_debug.loss(all_input_values)
        import numpy as np
        logger.record_tabular('[DEBUG] Distribution info:', np.exp(loss_debug[0][0]))
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
