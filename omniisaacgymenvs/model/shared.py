import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from omniisaacgymenvs.model.common import init_network
from omniisaacgymenvs.model.pointnet import PointNet
from omniisaacgymenvs.model.transformer_enc import TransformerEnc
from omniisaacgymenvs.algos.feature_extractor import PointCloudExtractor
from omniisaacgymenvs.model.attention import AttentionPooling

# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2,
                 reduction="sum", pcd_sampling_num = 500):
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
        
        self.pcd_sampling_num = pcd_sampling_num
        # self.pcd_backbone = init_network(self.pcd_config, input_channels=0, output_channels=[])
        self.pcd_backbone = PointNet()

        self.robot_state_num  = self.num_observations - pcd_sampling_num*3 - 3 # 3 is goal pos

        self.robot_state_mlp = nn.Sequential(nn.Linear(self.robot_state_num, 64),
                                             nn.ReLU(),
                                             nn.Linear(64, 64)) # comes from DexART

        # self.net = nn.Sequential(nn.Linear(256 + 64 + 3, 256),  # 256: pcd feature, 64: robot state feat, 3: goal pos
        #                          nn.ELU(),
        #                          nn.Linear(256, 128),
        #                          nn.ELU(),
        #                          nn.Linear(128, 64),
        #                          nn.ELU())
        
        self.net = nn.Sequential(nn.Linear(256 + 64 + 3, 64),  # 256: pcd feature, 64: robot state feat, 3: goal pos
                                 nn.ELU(),
                                 nn.Linear(64, 64),)

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
        RS: robot state,
        F: pcd feature dimension or robot state feature dimension,
        LF: linear projection feature dimension,
        C: number of sub-masks(channels)
        '''
        num_of_sub_mask = self.pcd_config["num_pcd_masks"] # TODO: 외부 config에서 pcd mask 개수를 받아와야 함.
        ### 240317 added. pcd mask 개수를 꼭 cfg로 받아야 하나? 가변적일 수도 있음. 우선 reaching target task에서는 1개로 고정.
        # TODO: goal pos는 robot state에 넣지 말고 combined features에 바로 붙여주자.
        N = self.pcd_sampling_num
        pcd_data = inputs["states"][:, :N*3]    # [B, N*3]. flatten pcd data
        pcd_pos_data = pcd_data.view([pcd_data.shape[0], -1, 3])    # [B, N, 3], 3 is x, y, z
        robot_state = inputs["states"][:, N*3:-3] # B, RS
        goal_pos = inputs["states"][:, -3:] # B, 3

        # robot state feature extractor
        robot_state_feature = self.robot_state_mlp(robot_state)  # [B, F_RS]

        # point cloud feature extractor
        ########################################## PointNet from DexART ##########################################
        point_feature = self.pcd_backbone(pcd_pos_data)  # [B, N, F_PCD]

        combined_features = torch.cat((point_feature, robot_state_feature, goal_pos), dim=1)  # [B, F_PCD+F_RS+3]

        ########################################## PointNet from DexART ##########################################

        # ### visualize tensorized pcd data
        # import open3d as o3d
        # env_pos = torch.tensor([[ 3.,  0.,  0.],
        #                         [ 0.,  0.,  0.],
        #                         [-3.,  0.,  0.]])
        # idx = 0
        # pcd = pcd_pos_data.detach().cpu().numpy()
        # pcd = pcd[0]    # visualize only first env
        
        # print(f'pcd shape: {pcd.shape}')
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(pcd)
        # o3d.visualization.draw_geometries([point_cloud],
        #                                     window_name=f'point cloud semantic {idx}')
        # ### visualize tensorized pcd data

    
        ########################################## PointNet2 ##########################################
        # # [B, N, 3] => [:, 3] , 3 is x, y, z.
        # # List point clouds without distinguishing between batches and pcd instances.
        # debathed_pcd_pos = pcd_pos_data.reshape([-1, 3])

        # # get the number of environments with a shape of [0, 1, 2, ...]
        # num_of_env = torch.arange(pcd_pos_data.shape[0], device=debathed_pcd_pos.device)

        # # repeat the number of environments for each point
        # batch = torch.repeat_interleave(num_of_env, N)

        # # get the point cloud feature
        # pointnet2_feature = self.pcd_backbone(None, debathed_pcd_pos, batch)  # [B, N, F]
        # point_feature = pointnet2_feature.mean(dim=1)  # [B, F]
        # # point_feature = F.adaptive_max_pool1d(pointnet2_feature.permute(0, 2, 1), 1).squeeze(2) # [B, F]

        # # # Concatenate along the third dimension
        # combined_features = torch.cat((point_feature, robot_state), dim=1)  # [B, F+RS]

        # # # gloabal max pooling (Dexpoint에서 쓰는 방식 적용)
        # # # 각 feature별로 max pooling을 한다. [B, N, F] => [B, F] (왜 이렇게 했을까....?)
        # # aaa = torch.max(point_feature, dim=1)[0]
        # # bbb = aaa.view(-1, 1, aaa.shape[-1]).repeat(1, N, 1)
        
        # # # concat local feats
        # # ccc = torch.cat([point_feature, bbb], dim=-1)

        # # # Output
        # # ddd = self.output_mlp(ccc)

        # # # Softmax
        # # output = torch.softmax(ddd, dim=-1)

        # # # Expand and repeat the robot states to match the dimensions
        # # robot_states_expanded = robot_state.unsqueeze(1).repeat(1, N, 1)

        # # point_features = F.adaptive_max_pool1d(point_feature.permute(0, 2, 1), 1).squeeze(2)

        # # # Concatenate along the third dimension
        # # combined_features = torch.cat((point_feature, robot_states_expanded), dim=2)
        ########################################## PointNet2 ##########################################




        if role == "policy":
            return self.mean_layer(self.net(combined_features)), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(self.net(combined_features)), {}