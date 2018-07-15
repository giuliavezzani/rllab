import time
import copy
import pickle
import numpy as np
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
#from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
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
            density_model_aux=None,
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
            iter_switch = None,
            normal_policy = True,
            env_type='rllab',
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

        ## This is the density model we use for compute the entropy.
        ## In this implementation the only option is a VAE (or VAE convolutional when images are used)
        self.density_model  = density_model

        ## This was a auxiliary VAE I used for maximizing the entropy at first on the entire state
        ## and then only on the latent variable. Those results were worst than the ones obtained only
        ## when maximizing the entropy on the latent variable since the beginning.
        self.density_model_aux = density_model_aux
        ## This clas contains the parameters for the density model
        self.args_density_model  = args_density_model
        ## This is actually useful, since only vae is available
        self.name_density_model  = name_density_model
        # Reward type can be: 'std', 'state_entropy' or 'policy_entropy'
        self.reward_type = reward_type
        # To decide on which part of the state to maximize entropy
        self.mask_state = mask_state
        # This was to decide if to fit VAE only on data collected at the curr_path_length
        # iteration (thus, on policy), or on a buffer (all the past data). Now this value
        # is fixed to 'no', since we only use the current data
        self.use_old_data = use_old_data
        # This might be useless
        self.network = network
        # To say the dimension of the latent variable
        self.bottleneck_size = bottleneck_size
        # The parameters of the shared layers of the latent variable
        self.file_network = file_network
        # Previously trained policy model
        self.file_model = file_model
        # This was the parameter to switch between the two vae (vae and vae_aux)
        self.iter_switch = iter_switch
        # Currently, this is fix equal to True
        self.normal_policy = normal_policy
        # This is deal with two different types of environment: rllab or gym env
        self.env_type = env_type

        if self.store_paths:
            logger.set_snapshot_dir(self.log_dir)

        # The vectorized sample is the sampler we use (actually for images tests we
        # hardcoded n_env=1 in the vectorizes_sample code, because vectorization seems to cause issues
        # in rendering)
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

    # This is the function to obtain samples for training
    def obtain_samples(self, itr, density_model, reward_type, name_density_model, mask_state, iter_switch=None, new_density_model=None, old_paths=None, normal_policy=True, env_type='rllab'):
        return self.sampler.obtain_samples(itr, density_model, reward_type, name_density_model, mask_state, iter_switch, new_density_model, old_paths, normal_policy, env_type)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths)

    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        # If a pre-trained policy is available
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
        # Passing some parameters to the vae
        self.density_model.scale = self.args_density_model.scale
        self.density_model.decay_entr = self.args_density_model.decay_entropy
        # This was for the auxiliary vae
        if self.density_model_aux is not None:
            self.density_model_aux.scale = self.args_density_model.scale
            self.density_model_aux.decay_entr = self.args_density_model.decay_entropy

        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")

                # This was in case the two vaes on different slices of the state were used
                if self.mask_state == "mix" or self.mask_state == "mix-nn":
                    if itr < self.iter_switch:
                        paths = self.obtain_samples(itr=itr, density_model=self.density_model_aux, reward_type=self.reward_type, name_density_model=self.name_density_model, mask_state=self.mask_state, iter_switch=self.iter_switch)
                    else:
                        paths = self.obtain_samples(itr=itr, density_model=self.density_model, reward_type=self.reward_type, name_density_model=self.name_density_model, mask_state=self.mask_state, iter_switch=self.iter_switch)
                else:
                    # This is for the other case, so the used one
                    paths = self.obtain_samples(itr=itr, density_model=self.density_model, reward_type=self.reward_type, name_density_model=self.name_density_model, mask_state=self.mask_state, iter_switch=self.iter_switch, normal_policy=self.normal_policy, env_type=self.env_type)


                logger.log("Processing samples...")
                samples_data = self.process_samples(itr, paths)
                logger.log("Logging diagnostics...")
                self.log_diagnostics(paths)


                # Psuedo count has been removed from the implementation
                #if not self.reward_type == 'pseudo-count':
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, samples_data)

                ### Train the density model every iteration
                if self.reward_type == 'state_entropy':
                    print('Training density model')
                    ## Using all state for hand-coded masks
                    #if self.network == None:
                    ## Mask state on all the objects for the block_vs_all_velocity
                    if (self.mask_state == "objects"):
                        self.mask_state_vect = np.zeros(14)
                        for l in range(14):
                            self.mask_state_vect[l] = 5 + l              ## All the object positions

                        self.mask_state_vect = self.mask_state_vect.astype(int)
                    ## Mask state on the object of interest
                    elif (self.mask_state == "one-object"):
                        self.mask_state_vect = np.zeros(2)
                        for l in range(2):
                            self.mask_state_vect[l] = 5 + l
                        self.mask_state_vect = self.mask_state_vect.astype(int)
                    ## This is for the ant environment
                    elif (self.mask_state == "com"):
                        self.mask_state_vect = np.zeros(2)
                        for l in range(0,2):
                            self.mask_state_vect[l] =  122 + l
                        self.mask_state_vect = self.mask_state_vect.astype(int)
                    ## Mask state on the pusher position
                    elif (self.mask_state == "pusher"):
                        self.mask_state_vect = np.zeros(2)
                        for l in range(2):
                            self.mask_state_vect[l] =  3 + l
                        self.mask_state_vect = self.mask_state_vect.astype(int)
                    ## Mask state on the pusher and object of interest positions
                    elif (self.mask_state == "pusher+object"):
                        self.mask_state_vect = np.zeros(4)
                        for l in range(2):
                            self.mask_state_vect[l] =   3 + l    # Pusher position
                        for l in range(2):
                            self.mask_state_vect[l+2] =  5 + l   # Object of interest position
                        self.mask_state_vect = self.mask_state_vect.astype(int)

                    ## Mask state on all the  other objects for the block_vs_all_velocity
                    elif (self.mask_state == "other-objects"):
                        self.mask_state_vect = np.zeros(12)

                        for l in range(12):
                            self.mask_state_vect[l] =  7 + l
                        self.mask_state_vect = self.mask_state_vect.astype(int)

                    ## This are the mask state for the toy environments
                    ## Mask on the coordinate of the object of interest that is important to move
                    elif (self.mask_state == "one-object-act"):
                        self.mask_state_vect = np.zeros(1)

                        self.mask_state_vect[0] =  1
                        self.mask_state_vect = self.mask_state_vect.astype(int)
                    ## Mask on the other part of the state
                    elif (self.mask_state == "other-object-act"):
                        self.mask_state_vect = np.zeros(1)

                        self.mask_state_vect[0] =  3
                        self.mask_state_vect = self.mask_state_vect.astype(int)

                    ## All the state
                    elif (self.mask_state == "all-act"):
                        self.mask_state_vect = np.zeros(4)

                        for l in range(4):
                            self.mask_state_vect[l] =  1 + 2*l
                        self.mask_state_vect = self.mask_state_vect.astype(int)

                    ## This is the option we suggest to use
                    if self.use_old_data == 'no':
                        samples_data_coll = []

                    ## Collecting samples when using a mask on the state
                    if (self.mask_state == "one-object-act" or self.mask_state == "all-act") or (self.mask_state == "other-object-act") or (self.mask_state == "objects") or (self.mask_state == "one-object") or (self.mask_state == "com") or (self.mask_state == "pusher") or (self.mask_state == "pusher+object") :
                        samples_data_coll.append([samples[self.mask_state_vect] for samples in samples_data['observations']])

                    ## Collecting samples when maximizing entropy all over the image
                    elif self.mask_state == 'images-all' or self.mask_state == 'images-estim':

                        if samples_data['images'].shape[0] > self.batch_size:
                            samples_data_coll.append(samples_data['images'][0:self.batch_size])

                        else:
                            samples_data_coll.append(samples_data['images'])

                    ## Collecting samples when maximizing the entropy over the entire state
                    else:

                        if samples_data['observations'].shape[0] > self.batch_size:
                            samples_data_coll.append(samples_data['observations'][0:self.batch_size])

                        else:
                            samples_data_coll.append(samples_data['observations'])

                    # Giving the samples to the vae
                    self.args_density_model.obs = samples_data_coll


                    self.args_density_model.itr = itr



                    print(np.asarray(samples_data_coll).ndim)
                    if (np.asarray(samples_data_coll).ndim> 1):

                        ## Training the vae
                        self.density_model.train(self.args_density_model, itr)

                        print('Density model trained')

                if np.mod(itr, self.gap) == 0:
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


                    ## Saving data
                    observations.append(samples_data['observations'])
                    #pickle.dump(observations, open(self.log_dir+'/observations.pkl', 'wb'))
                    rewards_real.append(samples_data['rewards_real'])

                    pickle.dump(rewards_real, open(self.log_dir+'/rewards_real.pkl', 'wb'))
                    rewards.append(samples_data['rewards'])
                    pickle.dump(rewards, open(self.log_dir+'/rewards.pkl', 'wb'))
                    returns.append(samples_data['returns'])
                    #pickle.dump(returns, open(self.log_dir+'/returns.pkl', 'wb'))

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
