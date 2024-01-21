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

        # self.task_embedding = nn.Parameter(torch.empty(1, 1, embed_dim))
        # TODO: task embedding 추가하기. embed_dim을 맞추어야 함.

        # point cloud feature extractor
        pcd_config = {
                      "model": "pn",
                      "dropout": 0.,
                      "num_pcd_masks": 2,       # TODO: 외부 config에서 pcd mask 개수를 받아와야 함.
                      "fps_deterministic": False,
                      "pos_in_feature": False,
                      "normalize_pos": True,
                      "normalize_pos_param": None
                      }
        
        for i in range(pcd_config["num_pcd_masks"]):
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
        '''
        B: batch size,
        N: number of points,
        F: feature dimension,
        C: number of sub-masks(channels)
        '''
        num_of_sub_mask = 2 # TODO: 외부 config에서 pcd mask 개수를 받아와야 함.
        N = 90  # number of sampled points
        pcd_data = inputs["states"] # [B, :]
        # TODO: 여기에서 pcd와 robot state를 구분. 뒤에서 25개가 robot state, 나머지는 pcd
        # TODO: Transformer에 넣어주기 위해서 robot state를 pcd feature 차원과 맞춰주어야 함. init에 linear layer를 넣어주어야 함.
        '''
        refer to observations in get_observations()
        dof_pos_scaled,                               # [NE, 6]
        dof_vel_scaled[:, :6] * generalization_noise, # [NE, 6]
        flange_pos,                                   # [NE, 3]
        flange_rot,                                   # [NE, 4]
        target_pos,                                   # [NE, 3]
        goal_pos,                                     # [NE, 3]
        total: 25
        '''

        point_feature = None
        pcd_pos_data = pcd_data.view([pcd_data.shape[0], -1, 3])    # [B, 90, 3], 3 is x, y, z


        '''
        # [:, 3] , 3 is x, y, z. List point clouds without distinguishing between batches and pcd instances.
        point_pos = torch.reshape(pcd_pos_data, (-1, pcd_pos_data.shape[-1])).to(dtype=torch.float32)

        # get the number of environments into [0, 1, 2, ...]
        num_of_env = torch.arange(pcd_pos_data.shape[0], device=point_pos.device)

        # repeat the number of environments for each point
        batch = torch.repeat_interleave(num_of_env, pcd_pos_data.shape[1], dim=0)
        '''


        '''
        ### visualize tensorized pcd data
        import open3d as o3d
        env_pos = torch.tensor([[ 3.,  0.,  0.],
                                [ 0.,  0.,  0.],
                                [-3.,  0.,  0.]])
        idx = 0
        idx_data = pcd_pos_data[:, 90*idx:90*(idx+1), :]
        pcd = idx_data.detach().cpu().numpy()
        pcd = pcd[0]    # visualize only first env
        
        
        print(f'pcd shape: {pcd.shape}')
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)
        o3d.visualization.draw_geometries([point_cloud],
                                            window_name=f'point cloud semantic {idx}')
        ### visualize tensorized pcd data
        '''

        # TODO: range(개수) 을 쓸게 아니라 state 또는 config를 통해 segmentation mask 개수를 받아올 것
        # TODO: sampling 개수를 90으로 적을게 아니라 state 또는 config를 통해 받아올 것
        # TODO: 참고사항: maniskill에서는 pcd에 mask를 씌워서 사용함. 이를 self-attention에 mask를 씌워서 필요한 정보만 연산되도록 함.
        point_features = None
        for i in range(num_of_sub_mask):
            pcd_backbone = getattr(self, f"pcd_backbone_{i}")

            
            # Get the each sub mask point cloud data via index
            sub_mask_pcd_pos = pcd_pos_data[:, N*i:N*(i+1), :]

            # [B, N, 3] => [:, 3] , 3 is x, y, z.
            # List point clouds without distinguishing between batches and pcd instances.
            pcd_pos = sub_mask_pcd_pos.reshape([-1, 3])

            # get the number of environments with a shape of [0, 1, 2, ...]
            num_of_env = torch.arange(pcd_pos_data.shape[0], device=pcd_pos.device)

            # repeat the number of environments for each point
            batch = torch.repeat_interleave(num_of_env, N)

            point_feature = pcd_backbone(None, pcd_pos, batch)  # [B, N, F]
            # TODO: 나중에 None 부분에 point feature로 vector field 등을 넣어주어야 함.
            # TODO: 현재는 submask별로 feature를 만들어서 concat해주고 있음. TF에 들어갈 때도 submask별로 들어가게 하기 위해 다른 mode를 만들어줘야 함.
            if point_features is None:
                point_features = point_feature
            else:
                point_features = torch.cat([point_features, point_feature], dim=1)
                # TODO: 지금 여기서 concat하면 batch 차원으로 쌓여짐... 그럼 transformer에 넣을 때 이상....
                # 원래 코드는 [환경 개수, observation 개수]로 넣어주고 있다.
                # 나는 [환경 개수, point 개수, point feature]로 넣어주거나 [환경 개수, point mask 개수, mask별 feature]로 넣어주어야 함.
                # TODO: point feature에 
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