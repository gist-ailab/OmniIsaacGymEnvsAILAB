import torch
import torch.nn as nn
import gym
import numpy as np

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from omniisaacgymenvs.model.transformer_enc import SharedTransformerEnc
# from omniisaacgymenvs.model.shared import Shared

# seed for reproducibility
seed = 42
set_seed(seed)  # e.g. `set_seed(42)` for fixed seed

env = load_omniverse_isaacgym_env(task_name="MovingTarget")
env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env.observation_space.shape[0],3), dtype=np.float32)
env = wrap_env(env)


device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = SharedTransformerEnc(env.observation_space, env.action_space, device)
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
Preprocessor가 필요할 경우, 위의 `RunningStandardScaler`를 참고하여 pcd에 대한 preprocessor을 수행해봐도 될 것 같다.
'''
# cfg["state_preprocessor"] = RunningStandardScaler   # 이 preprocessor는 pointcloud랑 맞진 않는다. 필요하면 나는 별도로 구성해야 할듯.
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# cfg["value_preprocessor"] = RunningStandardScaler
# cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 120
cfg["experiment"]["checkpoint_interval"] = 1200
cfg["experiment"]["directory"] = "runs/torch/MovingTarget"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 24000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()


# # ---------------------------------------------------------
# # comment the code above: `trainer.train()`, and...
# # uncomment the following lines to evaluate a trained agent
# # ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# # download the trained agent's checkpoint from Hugging Face Hub and load it
# path = download_model_from_huggingface("skrl/OmniIsaacGymEnvs-FrankaCabinet-PPO", filename="agent.pt")
# agent.load(path)

# # start evaluation
# trainer.eval()
