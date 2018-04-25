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

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]
            self.vec_env = VecEnvExecutor(
                envs=envs,
                max_path_length=self.algo.max_path_length
            )
        self.env_spec = self.algo.env.spec

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples(self, itr, density_model, reward_type, name_density_model, mask_state, new_density_model=None, old_paths=None):
        logger.log("Obtaining samples for iteration %d..." % itr)
        paths = []
        n_samples = 0
        obses = self.vec_env.reset()
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy
        import time
        while n_samples < self.algo.batch_size:
            #if old_paths == None:
            t = time.time()
            policy.reset(dones)
            actions, agent_infos = policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t
            obs_bottleneck = []


            ### Rewards for the point mass in exploration are 0.0
            scale = density_model.scale
            decay_entr = density_model.decay_entr

            if reward_type == 'state_entropy':
                #import IPython
                #IPython.embed()


                if (mask_state == "objects"):
                    self.mask_state_vect = np.zeros(14)
                    for l in range(14):
                        self.mask_state_vect[l] = 3 + l
                    self.mask_state_vect = self.mask_state_vect.astype(int)
                elif (mask_state == "one-object"):
                    self.mask_state_vect = np.zeros(2)
                    for l in range(2):
                        self.mask_state_vect[l] = 3 + l
                    self.mask_state_vect = self.mask_state_vect.astype(int)
                elif (mask_state == "com"):
                    self.mask_state_vect = np.zeros(2)
                    for l in range(0, 2):
                        self.mask_state_vect[l] =  l + 122
                    self.mask_state_vect = self.mask_state_vect.astype(int)


                rewards_real = np.zeros(shape=rewards.shape)


                #for l in range(len(next_obses)):
                if name_density_model == 'vae':
                    curr_noise = np.random.normal(size=(1, density_model.hidden_size))
                    if (mask_state == "objects" or mask_state == "one-object" or mask_state == "com"):
                        #if rewards[l] == -10:
                        rewards_real= rewards

                        rewards = rewards * scale +  [density_model.get_density(next_obs[self.mask_state_vect].reshape(1, next_obs[self.mask_state_vect].shape[0]), curr_noise) /((itr+1) ** decay_entr) for next_obs in next_obses]
                        #rewards = rewards * scale +  [density_model.get_density(next_obs[self.mask_state_vect].reshape(1, next_obs[self.mask_state_vect].shape[0]), curr_noise)  for next_obs in next_obses]
                        #elif rewards[l] == 0:
                            #rewards[l] += density_model.get_density(next_obses[l][self.mask_state_vect].reshape(1, next_obses[l][self.mask_state_vect].shape[0]), curr_noise) * scale
                    else:
                        #elif rewards[l] == 0:
                        rewards_real= rewards


                        ##### TODO: Temporary tests
                        #if itr < 90:
                        #    rewards =  [density_model.get_density(next_obs.reshape(1, next_obs.shape[0]), curr_noise) for next_obs in next_obses]
                        #else:
                        #    rewards = rewards

                        # TODO us this again:
                        rewards = rewards * scale + [density_model.get_density(next_obs.reshape(1, next_obs.shape[0]), curr_noise) /((itr+1) ** decay_entr) for next_obs in next_obses]
                        #print((itr+1)**decay_entr)
                        #rewards = rewards * scale + [density_model.get_density(next_obs.reshape(1, next_obs.shape[0]), curr_noise)  for next_obs in next_obses]


                else:

                    rewards = [density_model.get_density(next_obs.reshape(1, next_obs.shape[0])) for next_obs in next_obses]

            elif reward_type == 'policy_entropy':
                rewards_real = np.zeros(shape=rewards.shape)
                for l in range(len(next_obses)):
                    rewards_real[l]= rewards[l]
                    rewards[l] = rewards[l] * scale -np.log(policy.get_prob(actions[l], agent_infos['mean'][l], agent_infos['log_std'][l]))
            elif reward_type == 'discrete':
                self._dim_space = 20
                self._count_space = np.zeros(shape=(self._dim_space+1, self._dim_space+1))

                for l in range(len(next_obses)):

                    for h in range(0, self._dim_space+1):
                        for i in range(0, self._dim_space+1):

                            if next_obses[l][0] == h and next_obses[l][1] == i:
                                self._count_space[h,i] += 1

                h_spec = np.zeros(len(next_obses))
                i_spec = np.zeros(len(next_obses))
                for h in range(0, self._dim_space+1):
                    for i in range(0, self._dim_space+1):
                        for l in range(len(next_obses)):
                            if  next_obses[l][0] == h and  next_obses[l][1] == i:

                                h_spec[l] = h
                                i_spec[l] = i

                for l in range(len(next_obses)):
                    rewards[l] =  -self._count_space[int(h_spec[l]), int(i_spec[l])]/( len(next_obses))


            elif reward_type=='pseudo-count' and not new_density_model==None:
                for l in range(len(next_obses)):
                    curr_noise = np.random.normal(size=(1, density_model.hidden_size))
                    ro = np.exp(-density_model.get_density(next_obses[l].reshape(1, next_obses[l].shape[0]), curr_noise))
                    ro_new = np.exp(-new_density_model.get_density(next_obses[l].reshape(1, next_obses[l].shape[0]), curr_noise))
                    rewards[l] += (ro_new - ro) / ro * (1 - ro_new)

                #import IPython
                #IPython.embed()
            else:
                rewards_real = rewards
            #    for l in range(len(next_obses)):
            #        rewards[l] =  -np.linalg.norm(next_obses[l][0:2] - np.array([2.0, 2.0]))
            #print('reward', rewards)


            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
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
            #print(n_samples)

            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)


        return paths
