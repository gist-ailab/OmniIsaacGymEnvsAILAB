import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from omniisaacgymenvs.model.common import init_network
from omniisaacgymenvs.model.transformer_enc import TransformerEnc
from omniisaacgymenvs.algos.feature_extractor import PointCloudExtractor

# TODO: 여기에서는 point2를 통해 만든 feature를 받아서, 그것을 shared model에 넣어서 action을 만들어야 한다.

# define shared model (stochastic and deterministic models) using mixins
class SharedTransformerEnc(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, num_envs,
                 device, clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # point cloud feature extractor
        pcd_config = {"model": "pn",
                  "dropout": 0.,
                  "fps_deterministic": False,
                  "pos_in_feature": False,
                  "normalize_pos": True,
                  "normalize_pos_param": None}
        self.pcd_backbone = init_network(pcd_config, input_channels=0, output_channels=[])
        for i in range(num_envs):
            setattr(self, f"pcd_backbone_{i}", init_network(pcd_config, input_channels=0, output_channels=[]))
        # TODO: 나중에는 PointCloudExtractor를 사용하여 더 고도화된 feature extractor를 사용해야 함.
        # from omniisaacgymenvs.algos.feature_extractor import PointCloudExtractor
        

        self.net = TransformerEnc(
                                  input_dim=128,        # 이걸 pcd_config 넣어서 pcd랑 변수 맞추면 좋을 듯
                                  output_feature=64,    # 아래의 mean_layer, value_layer의 input dim과 변수 맞추면 좋을 듯
                                  )

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        # B: batch size, N: number of points, F: feature dimension
        pcd_data = inputs["states"] # [B, :]
        point_feature = None
        data = pcd_data.view([pcd_data.shape[0], -1, 3])    # [B, 90, N]
        point_pos = torch.reshape(data, (-1, data.shape[-1])).to(dtype=torch.float32)
        num_of_env = torch.arange(data.shape[0], device=point_pos.device)
        batch = torch.repeat_interleave(num_of_env, data.shape[1], dim=0)


        '''
        import open3d as o3d
        env_pos = torch.tensor([[ 3.,  0.,  0.],
                                [ 0.,  0.,  0.],
                                [-3.,  0.,  0.]])
        data_1 = data[:, 0:90, :]
        pcd = data_1.detach().cpu().numpy()
        pcd = pcd[0]
        idx = 0
        
        print(f'pcd shape: {pcd.shape}')
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)
        o3d.visualization.draw_geometries([point_cloud],
                                            window_name=f'point cloud semantic {idx}')
        '''

        # TODO: range(개수) 을 쓸게 아니라 state 또는 config를 통해 segmentation mask 개수를 받아올 것
        # TODO: sampling 개수를 90으로 적을게 아니라 state 또는 config를 통해 받아올 것
        # TODO: 참고사항: maniskill에서는 pcd에 mask를 씌워서 사용함. 이를 self-attention에 mask를 씌워서 필요한 정보만 연산되도록 함.
        point_features = None
        for i in range(2):
            pcd_backbone = getattr(self, f"pcd_backbone_{i}")
            batched_pcd_pos = data[:, 90*i:90*(i+1), :]
            pcd_pos = batched_pcd_pos.reshape([-1, 3])
            batch = torch.repeat_interleave(num_of_env, 90)
            point_feature = pcd_backbone(None, pcd_pos, batch)
            if point_features is None:
                point_features = point_feature
            else:
                point_features = torch.cat([point_features, point_feature], dim=0)
        # pointnet2 input dim:
        # 1: point_feature
        # 2: point_pos -> [num of points, 3]
        # 3: batch -> [num_of_point]
        # point_feature = self.pcd_backbone(point_feature, point_pos, batch)
        # TODO: n개의 point_feature를 concat해주어야 함. 이걸 transformer에 넣어 줌.
        # TODO: n개의 point_feature를 만드는 건, pcd_backbone에서 해주어야 함.
        if role == "policy":
            return self.mean_layer(self.net(point_features)), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(self.net(point_features)), {}