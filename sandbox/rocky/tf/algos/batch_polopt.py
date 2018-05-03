import time
import copy
import pickle
import numpy as np
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from rllab.sampler.utils import rollout

from sandbox.rocky.tf.networks.value_function import NNFunctionMultiHead


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            log_dir=None,
            gap=1,
            density_model=None,
            args_density_model= None,
            reward_type = 'std',
            name_density_model = 'vae',
            mask_state = 'all',
            use_old_data = 'yes',
            network = None,
            bottleneck_size = 16,
            graph = None,
            file_network = None,
            file_model = None,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.log_dir = log_dir
        self.gap = gap
        self.density_model  = density_model
        self.args_density_model  = args_density_model
        self.name_density_model  = name_density_model
        self.reward_type = reward_type
        self.mask_state = mask_state
        self.use_old_data = use_old_data
        self.network = network
        self.bottleneck_size = bottleneck_size
        self.file_network = file_network
        self.file_model = file_model

        if self.store_paths:
            logger.set_snapshot_dir(self.log_dir)

        if sampler_cls is None:
            # if self.policy.vectorized and not force_batch_sampler:
            sampler_cls = VectorizedSampler
            # else:
            #     sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)

        if self.file_model == None:
            self.init_opt()

    def start_worker(self):
        self.sampler.start_worker()

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, density_model, reward_type, name_density_model, mask_state, new_density_model=None, old_paths=None):
        return self.sampler.obtain_samples(itr, density_model, reward_type, name_density_model, mask_state,  new_density_model, old_paths)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths)

    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        if self.file_model is not None:
            import joblib
            self.policy = joblib.load(self.file_model)['policy']
            self.init_opt()
                # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = []
            for var in tf.global_variables():
                    # note - this is hacky, may be better way to do this in newer TF.
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.variables_initializer(uninit_vars))
        else:
            sess.run(tf.global_variables_initializer())

        self.start_worker()
        start_time = time.time()
        observations = []
        rewards = []
        rewards_real = []
        returns = []
        samples_data_coll = []
        self.density_model.scale = self.args_density_model.scale
        self.density_model.decay_entr = self.args_density_model.decay_entropy

        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")

                paths = self.obtain_samples(itr=itr, density_model=self.density_model, reward_type=self.reward_type, name_density_model=self.name_density_model, mask_state=self.mask_state)
                logger.log("Processing samples...")
                samples_data = self.process_samples(itr, paths)
                logger.log("Logging diagnostics...")
                self.log_diagnostics(paths)
                ### If we use pseudo-count the policy should updated later
                if not self.reward_type == 'pseudo-count':
                    logger.log("Optimizing policy...")
                    self.optimize_policy(itr, samples_data)

                ### Train the density model every iteration
                if self.reward_type == 'state_entropy' or self.reward_type == 'pseudo-count':
                    print('Training density model')
                    ## Using all state of hand-coded masks
                    if self.network == None:
                        if (self.mask_state == "objects"):
                            self.mask_state_vect = np.zeros(14)
                            for l in range(14):
                                self.mask_state_vect[l] = 3 + l

                            self.mask_state_vect = self.mask_state_vect.astype(int)
                        elif (self.mask_state == "one-object"):
                            self.mask_state_vect = np.zeros(2)
                            for l in range(2):
                                self.mask_state_vect[l] = 3 + l
                            self.mask_state_vect = self.mask_state_vect.astype(int)
                        elif (self.mask_state == "com"):
                            self.mask_state_vect = np.zeros(2)
                            for l in range(0,2):
                                self.mask_state_vect[l] =  122 + l
                            self.mask_state_vect = self.mask_state_vect.astype(int)


                        if self.use_old_data == 'no':
                            samples_data_coll = []
                        if (self.mask_state == "objects") or (self.mask_state == "one-object") or (self.mask_state == "com"):
                            samples_data_coll.append([samples[self.mask_state_vect] for samples in samples_data['observations']])
                        else:
                            ## TODO changes here
                            #if samples_data['observations'].shape[0] > 10000:
                            if samples_data['observations'].shape[0] > self.batch_size:
                                samples_data_coll.append(samples_data['observations'][0:self.batch_size])
                            #elif samples_data['observations'].shape[0] > 1000:

                                #samples_data_coll.append(samples_data['observations'][0:-1:5][0:1000])
                            else:
                                samples_data_coll.append(samples_data['observations'])
                    ## Using the learnt representation
                    else:
                        with self.sess.as_default():
                            with self.graph.as_default():
                                samples_data_coll.append(self.sess.run(self.network._output_for_shared(self.obs_pl, task=0, reuse=tf.AUTO_REUSE),
                                                                                    feed_dict={self.obs_pl: samples_data['observations']}))

                    self.args_density_model.obs = samples_data_coll

                    self.args_density_model.itr = itr

                    print(np.asarray(samples_data_coll).ndim)
                    if (np.asarray(samples_data_coll).ndim> 1):

                        ## TODO: Temporary: no reinit (no old data test)
                        ## Let's try to reinitialize everytime
                        #self.density_model.init_opt()
                        self.density_model.train(self.args_density_model, itr)

                        print('Density model trained')

################################ Pseudo count#################################################
                if self.reward_type == 'pseudo-count':
                    new_density = copy.copy(self.density_model)
                    paths = self.obtain_samples(itr=itr, density_model=self.density_model, reward_type=self.reward_type, name_density_model=self.name_density_model, mask_state=self.mask_state)
                    logger.log("Processing samples...")
                    samples_data = self.process_samples(itr, paths)
                    logger.log("Logging diagnostics...")
                    self.log_diagnostics(paths)
                    print('Training  NEW density model')
                    if (self.mask_state == "objects"):
                        self.mask_state_vect = np.zeros(14)
                        for l in range(14):
                            self.mask_state_vect[l] = 3 + l

                        self.mask_state_vect = self.mask_state_vect.astype(int)
                    elif (self.mask_state == "one-object"):
                        self.mask_state_vect = np.zeros(2)
                        for l in range(2):
                            self.mask_state_vect[l] = 3 + l
                        self.mask_state_vect = self.mask_state_vect.astype(int)
                    elif (self.mask_state == "com"):
                        self.mask_state_vect = np.zeros(2)
                        for l in range(0,2):
                            self.mask_state_vect[l] =  122 + l
                        self.mask_state_vect = self.mask_state_vect.astype(int)


                    if self.use_old_data == 'no':
                        samples_data_coll = []
                    if (self.mask_state == "objects") or (self.mask_state == "one-object") or (self.mask_state == "com"):
                        samples_data_coll.append([samples[self.mask_state_vect] for samples in samples_data['observations']])
                    else:
                        samples_data_coll.append(samples_data['observations'])
                    self.args_density_model.obs = samples_data_coll

                    self.args_density_model.itr = itr

                    ## Let's try to reinitialize everytime
                    new_density.init_opt()
                    new_density.train(self.args_density_model, itr)
                    print('NEW Density model trained')

                    paths = self.obtain_samples(itr=itr, density_model=self.density_model, reward_type=self.reward_type, name_density_model=self.name_density_model, mask_state=self.mask_state, new_density_model=new_density, old_paths=paths)
                    logger.log("Processing samples...")
                    samples_data = self.process_samples(itr, paths)
                    logger.log("Logging diagnostics...")
                    self.log_diagnostics(paths)
                    #print('rewards used', samples_data['rewards'])
                    ### If we use pseudo-count the policy should updated later
                    if not self.reward_type == 'pseudo-count':
                        logger.log("Optimizing policy...")
                        self.optimize_policy(itr, samples_data)
################################ Pseudo count#################################################

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                if self.store_paths:
                    logger.log("Saving params...")
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    rollout(self.env, self.policy, animated=True, max_path_length=self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")

                if np.mod(itr, self.gap) == 0:
                    observations.append(samples_data['observations'])
                    #pickle.dump(observations, open(self.log_dir+'/observations.pkl', 'wb'))
                    rewards_real.append(samples_data['rewards_real'])

                    pickle.dump(rewards_real, open(self.log_dir+'/rewards_real.pkl', 'wb'))
                    rewards.append(samples_data['rewards'])
                    pickle.dump(rewards, open(self.log_dir+'/rewards.pkl', 'wb'))
                    returns.append(samples_data['returns'])
                    pickle.dump(returns, open(self.log_dir+'/returns.pkl', 'wb'))



        self.shutdown_worker()
        if created_session:
            sess.close()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError
