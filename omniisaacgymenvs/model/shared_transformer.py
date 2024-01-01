import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from omniisaacgymenvs.model.common import init_network
from omniisaacgymenvs.model.transformer_enc import TransformerEncoder
from omniisaacgymenvs.algos.feature_extractor import PointCloudExtractor

# TODO: 여기에서는 point2를 통해 만든 feature를 받아서, 그것을 shared model에 넣어서 action을 만들어야 한다.

# define shared model (stochastic and deterministic models) using mixins
class SharedTransformerEnc(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # point cloud feature extractor
        config = {"model": "pn",
                  "dropout": 0.,
                  "fps_deterministic": False,
                  "pos_in_feature": False,
                  "normalize_pos": True,
                  "normalize_pos_param": None}
        self.pcd_backbone = init_network(config, input_channels=0, output_channels=[])

        self.net = TransformerEncoder(
                                      input_dim=128,        # 이걸 config에 넣어서 pcd랑 변수 맞추면 좋을 듯
                                      output_feature=64,    # 아래의 mean_layer, value_layer의 input dim과 변수 맞추면 좋을 듯
                                      )

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        # TODO: 여기에서는 point2를 통해 만든 feature를 받아서, 그것을 shared model에 넣어서 action을 만들어야 한다.
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        data = inputs["states"]
        point_feature = None
        point_pos = torch.reshape(data, (-1, data.shape[-1])).to(dtype=torch.float32)
        num_of_env = torch.arange(data.shape[0], device=point_pos.device)
        batch = torch.repeat_interleave(num_of_env, data.shape[1], dim=0)

        if role == "policy":
            point_feature = self.pcd_backbone(point_feature, point_pos, batch)
            return self.mean_layer(self.net(inputs["states"])), self.log_std_parameter, {}
        elif role == "value":
            point_feature = self.pcd_backbone(point_feature, point_pos, batch)
            return self.value_layer(self.net(inputs["states"])), {}