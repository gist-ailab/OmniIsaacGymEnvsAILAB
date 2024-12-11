import torch
import torch.nn as nn
import numpy as np
import os
from datetime import datetime
from copy import deepcopy

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer, StepTrainer
from skrl.utils import set_seed

import wandb

# from omniisaacgymenvs.model.shared_transformer import SharedTransformerEnc
from omniisaacgymenvs.model.shared import Shared

# seed for reproducibility
seed = 42
set_seed(seed)  # e.g. `set_seed(42)` for fixed seed

env = load_omniverse_isaacgym_env(task_name="PCDMovingTarget")
# env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env.observation_space.shape[0],3), dtype=np.float32)
env = wrap_env(env)

device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
# models["policy"] = SharedTransformerEnc(env.observation_space, env.action_space, env.num_envs, device)
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 16  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 8  # 16 * 4096 / 8192
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 5e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 2.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
'''
https://skrl.readthedocs.io/en/develop/api/resources/preprocessors/running_standard_scaler.html
https://skrl.readthedocs.io/en/develop/intro/getting_started.html#preprocessors
아래 전처리는 pointcloud랑 맞진 않는다. 필요하면 나는 별도로 구성해야 할듯.
Preprocessor가 필요할 경우, 위의 `RunningStandardScaler`를 참고하여 pcd에 대한 preprocessor을 수행해봐도 될 것 같다.
현재 observation space를 1차원으로 줄여서 아래 preprocessor를 사용해도 될듯? 방식은 한번 알아보자 
'''
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 10
cfg["experiment"]["checkpoint_interval"] = 100
# cfg["experiment"]["write_interval"] = 1
# cfg["experiment"]["checkpoint_interval"] = 4

now = datetime.now()
formatted_date = now.strftime("%y%m%d_%H%M%S")
cfg["experiment"]["directory"] = "runs/torch/PCDMovingTarget"
cfg["experiment"]["experiment_name"] = f"{formatted_date}_PCD_Moving_Target"
cfg["experiment"]["checkpoint_dir"] = "checkpoints"

cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"] = {
    "project": "ToolMani",
    "save_code": True,
    "sync_tensorboard": True,
    "name": f"{formatted_date}_PCD_Moving_Target_train",
    "tags": 't',
    "reinit": True
}

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)
# path = '/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/runs/torch/PCDMovingTarget/240516_224738_PCD_Moving_Target/checkpoints/agent_27500.pt'
# agent.load(path)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 30000, "headless": False}
trainer = StepTrainer(cfg=cfg_trainer, env=env, agents=agent)

if cfg["experiment"]["wandb_kwargs"]["save_code"]:
    wandb.save('/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/tasks/moving_target_pcd.py')

eval_cfg = deepcopy(cfg)
eval_cfg["experiment"]["write_interval"] = 1
eval_cfg["experiment"]["wandb_kwargs"] = {
    "project": "ToolMani",
    "sync_tensorboard": True,
    "name": f"{formatted_date}_PCD_Moving_Target_eval",
    "tags": 'e',
    "reinit": True
}
eval_interval = 300
assert eval_interval % eval_cfg["experiment"]["checkpoint_interval"] == 0, "eval_interval must be divisible by checkpoint_interval"
eval_memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)
eval_agent = PPO(models=models,
    memory=eval_memory,
    cfg=eval_cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device)
evaluater = StepTrainer(cfg=eval_cfg, env=env, agents=eval_agent)

for timestep in range(cfg_trainer["timesteps"]):
    trainer.train(timestep=timestep)
    if timestep % eval_interval == 0 and timestep > 0:


        
        '''그때그때 불러와서 interval만 바꾸어 평가하면 될듯'''

        path = '/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/runs/torch/PCDMovingTarget'
        # path = '/home/bak/.local/share/ov/pkg/isaac_sim-2023.1.1/' # debug 할 때 사용
        path = os.path.join(path,
                            eval_cfg["experiment"]["directory"],
                            eval_cfg["experiment"]["experiment_name"],
                            eval_cfg["experiment"]["checkpoint_dir"],
                            f'agent_{timestep}.pt')

        # path = os.path.join(path,                            
        #                     eval_cfg["experiment"]["experiment_name"],
        # )
        # path = os.path.join(path,
        #                     eval_cfg["experiment"]["checkpoint_dir"],)
        # path = os.path.join(path,
        #                     f'agent_{timestep}.pt')
        
        eval_agent.load(path)

        evaluater.eval(timestep=timestep)