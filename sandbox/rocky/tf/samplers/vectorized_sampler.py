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

    def obtain_samples(self, itr, density_model, reward_type, name_density_model):
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
            t = time.time()
            policy.reset(dones)
            actions, agent_infos = policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            ### Rewards for the point mass in exploration are 0.0

            if reward_type == 'state_entropy':
                #import IPython
                #IPython.embed()
                for l in range(len(next_obses)):
                    if name_density_model == 'vae':
                        curr_noise = np.random.normal(size=(1, density_model.hidden_size))
                        rewards[l] = density_model.get_density(next_obses[l].reshape(1, next_obses[l].shape[0]), curr_noise)
                    #else:
                        ##rewards[l] = density_model.get_density(next_obses[l].reshape(1, next_obses[l].shape[0]))

            elif reward_type == 'policy_entropy':
                for l in range(len(next_obses)):
                    rewards[l] = -np.log(policy.get_prob(actions[l], agent_infos['mean'][l], agent_infos['log_std'][l]))
            elif reward_type == 'discrete':
                self._dim_space = 100
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
                    rewards[l] =  self._count_space[int(h_spec[l]), int(i_spec[l])]/( len(next_obses))


            else:
                for l in range(len(next_obses)):
                    rewards[l] =  -np.linalg.norm(next_obses[l][0:2] - np.array([2.0, 2.0]))
            #print('reward', rewards)

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
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
                running_paths[idx]["rewards"].append(reward)
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
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        return paths
