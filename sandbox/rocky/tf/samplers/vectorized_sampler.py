import pickle

import tensorflow as tf
from rllab.sampler.base import BaseSampler
from sandbox.rocky.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools

from sandbox.rocky.tf.networks.value_function import NNFunctionMultiHead


class VectorizedSampler(BaseSampler):

    def __init__(self, algo, n_envs=None):
        super(VectorizedSampler, self).__init__(algo)
        self.n_envs = n_envs

    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

            ## NOTE: This is for the images, if you go from observations, remove it
            ## It will go faster
            n_envs = 1


        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]

            ## This is to adapt the sampler to a gym environment
            if hasattr(envs[0].wrapped_env, 'choice'):
                for l in range(len(envs)):
                    envs[l].wrapped_env.choice = self.algo.env.wrapped_env.choice
                    envs[l].wrapped_env.sparse = self.algo.env.wrapped_env.sparse
                    envs[l].wrapped_env.all_goals = self.algo.env.wrapped_env.all_goals
                    envs[l].wrapped_env.file_goals = self.algo.env.wrapped_env.file_goals
                    envs[l].wrapped_env.use_reward = self.algo.env.wrapped_env.use_reward
                    envs[l].wrapped_env.radius_reward = self.algo.env.wrapped_env.radius_reward

            self.vec_env = VecEnvExecutor(
                envs=envs,
                max_path_length=self.algo.max_path_length
            )

        self.env_spec = self.algo.env.spec

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples(self, itr, density_model, reward_type, name_density_model, mask_state, iter_switch, new_density_model=None, old_paths=None, normal_policy=True, env_type='rllab'):
        logger.log("Obtaining samples for iteration %d..." % itr)
        paths = []
        n_samples = 0

        obses = self.vec_env.reset()
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs
        if normal_policy is False:
            entropy = np.asarray(1.0 * self.vec_env.num_envs)

        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy

        import time
        while n_samples < self.algo.batch_size:

            t = time.time()
            policy.reset(dones)

            if normal_policy:
                actions, agent_infos = policy.get_actions(obses)
            else:
                actions, agent_infos = policy.get_actions_entropy(obses, entropy)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            ## NOTE: This is a temporary implementation. We use the images only for computing the bonus
            ## (if images-all or images-estim). The training is done from observations
            if mask_state == "images-all" or mask_state == "images-estim" :

                if env_type == 'rllab':
                    images = np.asarray(self.vec_env.get_images())
                else:
                    images = np.asarray(self.vec_env.get_images(env_type=env_type))

            env_time += time.time() - t
            obs_bottleneck = []

            scale = density_model.scale

            if reward_type == 'state_entropy':

                scale = density_model.scale
                decay_entr = density_model.decay_entr

                ## Here are all the same masks we explain in batch_polpt
                if (mask_state == "objects"):
                    self.mask_state_vect = np.zeros(14)
                    for l in range(14):
                        self.mask_state_vect[l] = 5 + l
                    self.mask_state_vect = self.mask_state_vect.astype(int)
                elif (mask_state == "one-object") or (mask_state == "mix"):
                    self.mask_state_vect = np.zeros(2)
                    for l in range(2):
                        self.mask_state_vect[l] = 5 + l
                    self.mask_state_vect = self.mask_state_vect.astype(int)
                elif (mask_state == "com"):
                    self.mask_state_vect = np.zeros(2)
                    for l in range(0, 2):
                        self.mask_state_vect[l] =  l + 122
                    self.mask_state_vect = self.mask_state_vect.astype(int)

                elif (mask_state == "pusher"):
                    self.mask_state_vect = np.zeros(2)
                    for l in range(2):
                        self.mask_state_vect[l] =  3 + l
                    self.mask_state_vect = self.mask_state_vect.astype(int)

                elif (mask_state == "pusher+object"):
                    self.mask_state_vect = np.zeros(4)
                    for l in range(2):
                        self.mask_state_vect[l] =   3 + l
                    for l in range(2):
                        self.mask_state_vect[l+2] =  5 + l
                    self.mask_state_vect = self.mask_state_vect.astype(int)

                elif (mask_state == "one-object-act"):
                    self.mask_state_vect = np.zeros(1)

                    self.mask_state_vect[0] =  1
                    self.mask_state_vect = self.mask_state_vect.astype(int)
                elif (mask_state == "other-object-act"):
                    self.mask_state_vect = np.zeros(1)

                    self.mask_state_vect[0] =  3
                    self.mask_state_vect = self.mask_state_vect.astype(int)

                elif (mask_state == "all-act"):
                    self.mask_state_vect = np.zeros(4)

                    for l in range(4):
                        self.mask_state_vect[l] =  1 + 2*l
                    self.mask_state_vect = self.mask_state_vect.astype(int)


                rewards_real = np.zeros(shape=rewards.shape)

                ## Here we augment the reward (sparse) with a bonus, computed from vae_conv
                ## density_model.get_density provide an estimate of -log(p(z)), where z is the variable
                ## on which we  want to maximize the entropy
                if name_density_model == 'vae':
                    ## Noise for the vae
                    curr_noise = np.random.normal(size=(1, density_model.hidden_size))
                    if (mask_state == "one-object-act" or mask_state == "other-object-act" or mask_state == "all-act" or mask_state == "objects" or mask_state == "one-object" or mask_state == "com" or mask_state == "pusher" or mask_state == "pusher+object"):
                        ## This are the real rewards output by the environment
                        rewards_real= rewards
                        ## The real rewards are scaled with a scale factor, to encourage to increase the real rewards, one experience
                        rewards = rewards * scale +  [density_model.get_density(next_obs[self.mask_state_vect].reshape(1, next_obs[self.mask_state_vect].shape[0]), curr_noise) /((itr+1) ** decay_entr) for next_obs in next_obses]
                        entropy =  [100 * density_model.get_density(next_obs[self.mask_state_vect].reshape(1, next_obs[self.mask_state_vect].shape[0]), curr_noise) /((itr+1) ** decay_entr) for next_obs in next_obses]
                    elif (mask_state == "mix"):
                        if itr < iter_switch:
                            rewards_real= rewards
                            rewards = rewards * scale + [density_model.get_density(next_obs.reshape(1, next_obs.shape[0]), curr_noise) /((itr+1) ** decay_entr) for next_obs in next_obses]
                            entropy =  [100 * density_model.get_density(next_obs[self.mask_state_vect].reshape(1, next_obs[self.mask_state_vect].shape[0]), curr_noise) /((itr+1) ** decay_entr) for next_obs in next_obses]
                        else:
                            rewards_real= rewards

                            rewards = rewards * scale +  [density_model.get_density(next_obs[self.mask_state_vect].reshape(1, next_obs[self.mask_state_vect].shape[0]), curr_noise) /((itr+1) ** decay_entr) for next_obs in next_obses]
                            entropy = [100 * density_model.get_density(next_obs[self.mask_state_vect].reshape(1, next_obs[self.mask_state_vect].shape[0]), curr_noise) /((itr+1) ** decay_entr) for next_obs in next_obses]

                    elif  mask_state=="images-all":

                        rewards = rewards * scale + [density_model.get_density(image.reshape(1, image.shape[0], image.shape[1], image.shape[2]), curr_noise) /((itr+1) ** decay_entr) for image in images]
                    elif mask_state=="images-estim" :
                        rewards = rewards * scale + [density_model.get_density(image.reshape(1, image.shape[0], image.shape[1], image.shape[2]), curr_noise) /((itr+1) ** decay_entr) for image in images]

                    else:
                        rewards_real= rewards
                        rewards = rewards * scale + [density_model.get_density(next_obs.reshape(1, next_obs.shape[0]), curr_noise) /((itr+1) ** decay_entr) for next_obs in next_obses]

                else:
                    rewards = [density_model.get_density(next_obs.reshape(1, next_obs.shape[0])) for next_obs in next_obses]


            elif reward_type == 'policy_entropy':
                ## Here we maximize the entropy over the actions
                rewards_real = np.zeros(shape=rewards.shape)
                for l in range(len(next_obses)):
                    rewards_real[l]= rewards[l]
                    rewards[l] = rewards[l] * scale -np.log(policy.get_prob(actions[l], agent_infos['mean'][l], agent_infos['log_std'][l]))

            else:
                rewards_real = rewards

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]

            if not (mask_state =='images-all' or mask_state == 'images-estim'):
                for idx, observation, action, reward, env_info, agent_info, done, r_real in zip(itertools.count(), obses, actions,
                                                                                        rewards, env_infos, agent_infos,
                                                                                        dones, rewards_real):
                    if running_paths[idx] is None:
                        running_paths[idx] = dict(
                            observations=[],
                            actions=[],
                            rewards=[],
                            env_infos=[],
                            agent_infos=[],
                            rewards_real=[],
                        )
                    running_paths[idx]["observations"].append(observation)
                    running_paths[idx]["actions"].append(action)
                    running_paths[idx]["rewards"].append(reward)
                    running_paths[idx]["rewards_real"].append(r_real)
                    running_paths[idx]["env_infos"].append(env_info)
                    running_paths[idx]["agent_infos"].append(agent_info)

                    if done:
                        paths.append(dict(
                            observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                            actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                            rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                            env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                            agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                            rewards_real=tensor_utils.stack_tensor_list(running_paths[idx]["rewards_real"]),
                        ))
                        n_samples += len(running_paths[idx]["rewards"])
                        running_paths[idx] = None

                process_time += time.time() - t
                pbar.inc(len(obses))
                obses = next_obses
            else:

                for idx, observation, action, reward, env_info, agent_info, done, r_real, img in zip(itertools.count(), obses, actions,
                                                                                        rewards, env_infos, agent_infos,
                                                                                        dones, rewards_real, images):
                    if running_paths[idx] is None:
                        running_paths[idx] = dict(
                            observations=[],
                            actions=[],
                            rewards=[],
                            env_infos=[],
                            agent_infos=[],
                            rewards_real=[],
                            images=[],
                        )
                    running_paths[idx]["observations"].append(observation)
                    running_paths[idx]["actions"].append(action)
                    running_paths[idx]["rewards"].append(reward)
                    running_paths[idx]["rewards_real"].append(r_real)
                    running_paths[idx]["env_infos"].append(env_info)
                    running_paths[idx]["agent_infos"].append(agent_info)
                    running_paths[idx]["images"].append(img)


                    if done:
                        paths.append(dict(
                            observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                            actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                            rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                            env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                            agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                            rewards_real=tensor_utils.stack_tensor_list(running_paths[idx]["rewards_real"]),
                            images=tensor_utils.stack_tensor_list(running_paths[idx]["images"]),
                        ))
                        n_samples += len(running_paths[idx]["rewards"])
                        running_paths[idx] = None


                process_time += time.time() - t
                pbar.inc(len(obses))
                obses = next_obses



        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)


        return paths
