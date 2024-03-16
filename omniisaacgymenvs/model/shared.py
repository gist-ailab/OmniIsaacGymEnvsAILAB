import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from omniisaacgymenvs.model.common import init_network
from omniisaacgymenvs.model.transformer_enc import TransformerEnc
from omniisaacgymenvs.algos.feature_extractor import PointCloudExtractor
from omniisaacgymenvs.model.attention import AttentionPooling

# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)


        # point cloud feature extractor
        self.pcd_config = {
                           "model": "pn",
                           "dropout": 0.,
                        #    "num_pcd_masks": 2,       # TODO: 외부 config에서 pcd mask 개수를 받아와야 함.
                           "num_pcd_masks": 1,       # TODO: 외부 config에서 pcd mask 개수를 받아와야 함.
                           "fps_deterministic": False,
                           "pos_in_feature": False,
                           "normalize_pos": True,
                           "normalize_pos_param": None,
                           "attention_num_heads": 8,
                           "attention_hidden_dim": 64,
                      }
        
        pcd_backbone = init_network(self.pcd_config, input_channels=0, output_channels=[])
        # TODO: pcd backbone으로 pcd feature를 만듦.
        # TODO: robot state 및 환경 state 만 따로 뽑아서 MLP로 feature 생성
        # TODO: 위 두개를 concat해서 mean layer와 value layer에 통과시켜 policy, value로 사용

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            return self.mean_layer(self.net(inputs["states"])), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(self.net(inputs["states"])), {}