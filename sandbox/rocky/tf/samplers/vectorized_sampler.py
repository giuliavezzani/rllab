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

    def obtain_samples(self, itr, density_model, reward_type, name_density_model, reward=False):
        logger.log("Obtaining samples for iteration %d..." % itr)
        paths = []
        n_samples = 0
        if reward == False:
            self.obses = self.vec_env.reset()
        self.dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0
        count = 0

        if reward == False:
            self.obses_coll = []
            self.dones_coll = []
            self.actions_coll = []
            self.obses_coll = []
            self.env_infos_coll = []
            self.agent_infos_coll = []

        policy = self.algo.policy
        import time
        while n_samples < self.algo.batch_size:

            t = time.time()
            if reward == False:
                policy.reset(self.dones)
                self.actions, self.agent_info = policy.get_actions(self.obses)
                self.actions_coll.append(self.actions)
                #self.agent_infos_coll.append(self.agent_info)

            policy_time += time.time() - t
            t = time.time()

            #### Situation is different if you we use state entropy
            if reward_type == 'state_entropy':
                # At first we collect the data just to get observations
                if reward == False:
                    self.next_obses, rewards, self.dones, self.env_info = self.vec_env.step(self.actions)
                    env_time += time.time() - t

                    self.obses_coll.append(self.next_obses)
                    self.dones_coll.append(self.dones)
                    #self.env_infos_coll.append(self.env_info)
                    ## Here the function return the paths directly
                else:
                    ## Uses the observations and everything collected when reward == False
                    env_time += time.time() - t
                    rewards = np.zeros(len(self.next_obses))


                    ## Compute the rewards with the trained vae
                    for l in range(len(self.obses_coll[count])):
                        if name_density_model == 'vae':
                            #import IPython
                            #IPython.embed()
                            curr_noise = np.random.normal(size=(1, density_model.hidden_size))
    
                            rewards[l] = density_model.get_density(self.obses_coll[count][l][density_model.starting_state:-1].reshape(1, self.obses_coll[count][l].shape[0] - (density_model.starting_state+1)), curr_noise)
                        else:
                            rewards[l] = -density_model.get_density(self.obses_coll[count][l].reshape(1, self.obses_coll[count][l].shape[0]))

                    t = time.time()

            else:
                ### Here is everything regular
                self.next_obses, rewards, self.dones, self.env_info = self.vec_env.step(self.actions)
                env_time += time.time() - t

                self.obses_coll.append(self.next_obses)
                self.dones_coll.append(self.dones)
                self.env_infos_coll.append(self.env_info)


                if reward_type == 'policy_entropy':
                    for l in range(len(self.next_obses)):
                        rewards[l] = -np.log(policy.get_prob(self.actions[l], self.agent_info['mean'][l], self.agent_info['log_std'][l]))
                elif reward_type == 'discrete':
                    self._dim_space = 20
                    self._count_space = np.zeros(shape=(self._dim_space+1, self._dim_space+1))

                    for l in range(len(self.next_obses)):

                        for h in range(0, self._dim_space+1):
                            for i in range(0, self._dim_space+1):

                                if self.next_obses[l][0] == h and self.next_obses[l][1] == i:
                                    self._count_space[h,i] += 1

                    h_spec = np.zeros(len(self.next_obses))
                    i_spec = np.zeros(len(self.next_obses))
                    for h in range(0, self._dim_space+1):
                        for i in range(0, self._dim_space+1):
                            for l in range(len(self.next_obses)):
                                if  self.next_obses[l][0] == h and  self.next_obses[l][1] == i:

                                    h_spec[l] = h
                                    i_spec[l] = i

                    for l in range(len(self.next_obses)):
                        rewards[l] =  -self._count_space[int(h_spec[l]), int(i_spec[l])]/( len(self.next_obses))


                else:
                    for l in range(len(self.next_obses)):
                        rewards[l] =  -np.linalg.norm(self.next_obses[l][0:2] - np.array([2.0, 2.0]))
                #print('reward', rewards)

            t = time.time()

            #Return everything
            if reward == False:
                self.agent_infos = tensor_utils.split_tensor_dict_list(self.agent_info)
                self.env_infos = tensor_utils.split_tensor_dict_list(self.env_info)
                if self.env_infos is None:
                    self.env_infos = [dict() for _ in range(self.vec_env.num_envs)]
                if self.agent_infos is None:
                    self.agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
                self.agent_infos_coll.append(self.agent_infos)
                self.env_infos_coll.append(self.env_infos)
            for idx, observation, action, reward_v, env_info, agent_info, done in zip(itertools.count(), self.obses_coll[count], self.actions_coll[count],
                                                                                    rewards, self.env_infos_coll[count], self.agent_infos_coll[count],
                                                                                    self.dones_coll[count]):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )


                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward_v)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

                if done:
                    paths.append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None

            process_time += time.time() - t
            pbar.inc(len(self.obses))
            self.obses = self.next_obses
            count += 1

        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        return paths
