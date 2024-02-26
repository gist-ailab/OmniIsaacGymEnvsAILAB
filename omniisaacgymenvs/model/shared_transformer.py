import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from omniisaacgymenvs.model.common import init_network
from omniisaacgymenvs.model.transformer_enc import TransformerEnc
from omniisaacgymenvs.algos.feature_extractor import PointCloudExtractor
from omniisaacgymenvs.model.attention import AttentionPooling

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
        
        for i in range(self.pcd_config["num_pcd_masks"]):
            setattr(self, f"pcd_backbone_{i}", init_network(self.pcd_config, input_channels=0, output_channels=[]))
        # TODO: 나중에는 PointCloudExtractor를 사용하여 더 고도화된 feature extractor를 사용해야 함.
        # from omniisaacgymenvs.algos.feature_extractor import PointCloudExtractor
        
        transformer_input_dim = 128
        projection_output_dim = 64

        self.combine_pcd_robot_state = True

        # self.robot_state_layer = nn.Linear(25, transformer_input_dim) # TODO: robot state개수 25를 config에서 받아와야 함.
        self.robot_state_layer = nn.Linear(22, transformer_input_dim) # reaching target 하려고 임시로 넣어둔 것. 나중에는 이 부분을 지워야 함.
        self.transformer_enc = TransformerEnc(
                                              input_dim=transformer_input_dim,  # 이걸 pcd_config 넣어서 pcd랑 변수 맞추면 좋을 듯
                                              #   output_feature=transformer_output_dim,    # 아래의 mean_layer, value_layer의 input dim과 변수 맞추면 좋을 듯
                                              )
        
        self.attention_pooling = AttentionPooling(embed_dim=transformer_input_dim,
                                                  num_heads=self.pcd_config['attention_num_heads'],
                                                  latent_dim=self.pcd_config['attention_hidden_dim'])


        self.linear_projection = nn.Linear(transformer_input_dim, projection_output_dim)

        self.att_linear_projection = nn.Sequential(
                                                   AttentionPooling(embed_dim=transformer_input_dim,
                                                                    num_heads=self.pcd_config['attention_num_heads'],
                                                                    latent_dim=self.pcd_config['attention_hidden_dim']),
                                                   nn.ELU(),
                                                   nn.Linear(transformer_input_dim, projection_output_dim),
                                                   nn.ELU(),
                                                  )

        self.mean_layer = nn.Linear(projection_output_dim, self.num_actions)
        self.value_layer = nn.Linear(projection_output_dim, 1)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        '''
        B: batch size,
        N: number of points,
        RS: robot state,
        F: pcd feature dimension or robot state feature dimension,
        LF: linear projection feature dimension,
        C: number of sub-masks(channels)
        '''
        num_of_sub_mask = self.pcd_config["num_pcd_masks"] # TODO: 외부 config에서 pcd mask 개수를 받아와야 함.
        N = 100  # number of sampled points
        observations = inputs["states"] # [B, N*3 + RS], 3 for x, y, z
        # pcd_data = observations[:, :-25] # [B, N*3]
        # robot_state = observations[:, -25:] # [B, RS]
        pcd_data = observations[:, :-22] # reaching target 하려고 임시로 넣어둔 것. 나중에는 이 부분을 지워야 함.
        robot_state = observations[:, -22:] # reaching target 하려고 임시로 넣어둔 것. 나중에는 이 부분을 지워야 함.
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
        # TODO: pcd_feature만 transformer에 넣어주는 것과, pcd_feature와 robot state를 concat해서 transformer에 넣어주는 것을 구분해서 실험해보자.
        if self.combine_pcd_robot_state:
            # combine pointnet2 feature and robot state feature
            robot_state_feature = self.robot_state_layer(robot_state)   # [B, RS] => [B, F]
            robot_state_feature = robot_state_feature.unsqueeze(1)      # [B, F] => [B, 1, F]
            combined_feature = torch.cat([point_features, robot_state_feature], dim=1)   # [B, N, F] + [B, 1, F] => [B, N+1, F]

            # input the combined feature to transformer
            pcd_robot_feature = self.transformer_enc(combined_feature)    # [B, N+1, F] => [B, N+1, F]
        else:
            # input the pointnet2 feature to transformer
            point_transformer_features = self.transformer_enc(point_features)   # [B, N, F] => [B, N, F]

            # combine pcd transformer feature and robot state feature
            robot_state_feature = self.robot_state_layer(robot_state)   # [B, RS] => [B, F]
            robot_state_feature = robot_state_feature.unsqueeze(1)      # [B, F] => [B, 1, F]
            pcd_robot_feature = torch.cat([point_transformer_features, robot_state_feature], dim=1)  # [B, N, F] + [B, 1, F] => [B, N+1, F]
        
        # attention pooling
        linear_feature = self.att_linear_projection(pcd_robot_feature)  # [B, N+1, F] => [B, LF]
        
        if role == "policy":
            return self.mean_layer(linear_feature), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(linear_feature), {}