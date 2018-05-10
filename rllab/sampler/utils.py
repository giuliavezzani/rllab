import numpy as np
from rllab.misc import tensor_utils
import time


def rollout(env, agent,  max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    images = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:

        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        #com = env.get_body_com("torso")
        # ref_x = x + self._init_torso_x
        #print('pos to goal ', np.sum(np.abs(com[:2] - env.goals[env._goal_idx])))
        #print('real reward', r)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1

        if d:
            break
        o = next_o
        if animated:
            env.render()
            data, width, height = env.wrapped_env.wrapped_env.get_viewer().get_image()
            images.append(np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:])
            timestep = 0.05
            time.sleep(timestep / speedup)

    #if animated and not always_return_paths:
    #    return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        images=tensor_utils.stack_tensor_list(images),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
