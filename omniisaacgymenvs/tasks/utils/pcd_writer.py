# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional
import torch
import torch.utils.dlpack
import carb
import warp as wp
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omniisaacgymenvs.tasks.utils.pcd_listener import PointcloudListener

import os
import open3d as o3d
import open3d.core as o3c
import numpy as np
import point_cloud_utils as pcu
import copy

class PointcloudWriter(Writer):
    """A custom writer that uses omni.replicator API to retrieve RGB data via render products
        and formats them as tensor batches. The writer takes a PytorchListener which is able
        to retrieve pytorch tensors for the user directly after each writer call.

    Args:
        listener (PoincloudListener): A PoincloudListener that is sent pytorch batch tensors at each write() call.
        output_dir (str): directory in which rgb data will be saved in PNG format by the backend dispatch.
                          If not specified, the writer will not write rgb data as png and only ping the
                          listener with batched tensors.
        device (str): device in which the pytorch tensor data will reside. Can be "cpu", "cuda", or any
                      other format that pytorch supports for devices. Default is "cuda".
    """

    def __init__(self, listener: PointcloudListener,
                 output_dir: str = None,
                 pcd_sampling_num: int = 100,
                 pcd_normalize: bool = False,
                 env_pos: torch.Tensor = None,
                 camera_positions: dict = None,
                 camera_orientations: tuple = None,
                 visualize_point_cloud: bool = False,
                 device: str = "cuda"):
        # If output directory is specified, writer will write annotated data to the given directory
        if output_dir:
            self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
            self._backend = self.backend
            self._output_dir = self.backend.output_dir
        else:
            self._output_dir = None
        self._frame_id = 0

        self.annotators = [AnnotatorRegistry.get_annotator("pointcloud")]
        self.listener = listener
        self.pcd_sampling_num = pcd_sampling_num
        self.pcd_normalize = pcd_normalize
        self.env_pos = env_pos
        self.camera_positions = camera_positions
        self.camera_orientations = camera_orientations
        self.visualize_point_cloud = visualize_point_cloud
        self.device = device

    def write(self, data: dict) -> None:
        """Sends data captured by the attached render products to the PytorchListener and will write data to
        the output directory if specified during initialization.

        Args:
            data (dict): Data to be pinged to the listener and written to the output directory if specified.
        """
        if self._output_dir:
            # Write RGB data to output directory as png
            self._write_pcd(data)   # 이부분은 point cloud 저장하고 싶을 때 따로 함수 변경하자.
        # pytorch_rgb = self._convert_to_pytorch(data).to(self.device)
        pointcloud = self._convert_to_pointcloud(data)
        self.listener.write_data(pointcloud)
        self._frame_id += 1

    @carb.profiler.profile
    def _write_rgb(self, data: dict) -> None:
        for annotator in data.keys():
            if annotator.startswith("LdrColor"):
                render_product_name = annotator.split("-")[-1]
                file_path = f"rgb_{self._frame_id}_{render_product_name}.png"
                img_data = data[annotator]
                if isinstance(img_data, wp.types.array):
                    img_data = img_data.numpy()
                self._backend.write_image(file_path, img_data)

    @carb.profiler.profile
    def _write_pcd(self, data: dict) -> None:
        # TODO: ply로 저장해서 pointcloud를 확인해보고 싶다면 이 함수를 수정
        ply_out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "out")
        os.makedirs(ply_out_dir, exist_ok=True)

        for annotator in data.keys():
            if annotator.startswith("pointcloud"):
                pcd = o3d.geometry.PointCloud()
                render_product_name = annotator.split("-")[-1]
                file_path = f"pcd_{self._frame_id}_{render_product_name}.ply"
                pcd_data = data[annotator]
                pcd.points = o3d.utility.Vector3dVector(pcd_data)
                pcd.colors = o3d.utility.Vector3dVector(pcd_data)
                if isinstance(pcd_data, wp.types.array):
                    pcd_data = pcd_data.numpy()
                o3d.io.write_point_cloud(file_path, pcd)
                self._backend.write_image(file_path, pcd_data)

    @carb.profiler.profile
    def _convert_to_pytorch(self, data: dict) -> torch.Tensor:
        # 이 함수는 이미지를 torch tensor로 바꿔주는 함수
        if data is None:
            raise Exception("Data is Null")

        data_tensors = []
        for annotator in data.keys():
            if annotator.startswith("LdrColor"):
                data_tensors.append(wp.to_torch(data[annotator]).unsqueeze(0))
                # data[annotator] 데이터 형식은 Nvidia의 warp 형식이다. 이를 pytorch 형식으로 바꿔준다.

        # Move all tensors to the same device for concatenation
        device = "cuda:0" if self.device == "cuda" else self.device
        data_tensors = [t.to(device) for t in data_tensors]

        data_tensor = torch.cat(data_tensors, dim=0)
        return data_tensor
    
    @carb.profiler.profile
    def _convert_to_pointcloud(self, data: dict) -> torch.Tensor:
        if data is None:
            raise Exception("Data is Null")

        # TODO: for문 말고 matrix화 시킬 수 있다면 한번 시도를....
        num_samples = self.pcd_sampling_num

        previous_env_index = 0
        for annotator in data.keys():
            if annotator.startswith("pointcloud"):
                if len(annotator.split('_'))==2:
                    # idx = f'env_{0}'
                    idx = 0
                elif len(annotator.split('_'))==3:
                    # idx = f'env_{int(annotator.split("_")[-1])}'
                    idx = int(annotator.split("_")[-1]) # env_n

                env_index = idx//3
                env_center = self.env_pos[env_index].to(self.device)
                
                camera_idx = idx%3

                pcd_pos = torch.from_numpy(data[annotator]['data']).to(self.device)
                pcd_normal = torch.from_numpy(data[annotator]['info']['pointNormals']).to(self.device)
                pcd_semantic = torch.from_numpy(data[annotator]['info']['pointSemantic']).to(self.device)

                pcd_pos = pcd_pos.unsqueeze(0)
                pcd_normal = pcd_normal.unsqueeze(0)
                pcd_semantic = pcd_semantic.unsqueeze(0)

                if pcd_pos.shape[1]==0:
                    pass
                elif camera_idx == 0:
                    each_env_pcd_pos = pcd_pos
                    each_env_pcd_normal = pcd_normal
                    each_env_pcd_semantic = pcd_semantic
                else:
                    try:
                        each_env_pcd_pos = torch.cat((each_env_pcd_pos, pcd_pos), dim=1)
                        each_env_pcd_normal = torch.cat((each_env_pcd_normal, pcd_normal), dim=1)
                        each_env_pcd_semantic = torch.cat((each_env_pcd_semantic, pcd_semantic), dim=1)
                    except:
                        each_env_pcd_pos = pcd_pos
                        each_env_pcd_normal = pcd_normal
                        each_env_pcd_semantic = pcd_semantic

                if camera_idx == 2:
                    centerized_pcd_pos = torch.sub(each_env_pcd_pos, env_center)

                    sampled_pcd_pos = self._sampling_pcd(
                                                         idx,
                                                         centerized_pcd_pos.squeeze(0),
                                                         each_env_pcd_semantic.squeeze(0),
                                                         num_samples,
                                                         self.pcd_normalize,
                                                         )
                    sampled_pcd_pos = sampled_pcd_pos.unsqueeze(0)
                    if env_index == 0:
                        pcd_pos_tensors = sampled_pcd_pos
                    else:
                        pcd_pos_tensors = torch.cat((pcd_pos_tensors, sampled_pcd_pos), dim=0)

                    '''
                    pcd_np = sampled_pcd_pos.squeeze(0).detach().cpu().numpy()
                    
                    
                    print(f'pcd shape: {pcd_np.shape}')
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(pcd_np)
                    o3d.visualization.draw_geometries([point_cloud],
                                                        window_name=f'point cloud semantic {idx}')
                    '''
                    


        return pcd_pos_tensors

    def _sampling_pcd(self,        # also conduct sampling
                      idx: int,
                      pcd_pos,
                      pcd_semantic,
                      num_samples: int,
                      normalize: bool,
                      ) -> torch.Tensor:
        semantics = torch.unique(pcd_semantic)
        # print(f'semantics: {semantics}')

        # sampling point cloud by semantic
        for idx in range(semantics.shape[0]):
            # print(f'idx: {idx}, semantic: {semantics[idx]}')
            index = torch.unique(pcd_semantic)[idx]
            pcd_idx = torch.where(pcd_semantic==index)[0]
            pcd = pcd_pos[pcd_idx]

            # device_num = torch.cuda.current_device()
            # device = o3d.core.Device(f"{self.device}:{device_num}")

            if pcd.shape[0] >= num_samples: # point cloud down sampling
                # 일반적인 o3d는 cpu에 올려져 있는데, tensor로 바꿔주기 위헤 아래와 같이 타입을 바꿔준다.
                o3d_tensor = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pcd))
                o3d_pcd_tensor = o3d.t.geometry.PointCloud(o3d_tensor)
                o3d_downsampled_pcd = o3d_pcd_tensor.farthest_point_down_sample(num_samples)
                downsampled_points = o3d_downsampled_pcd.point
                torch_tensor_points = torch.utils.dlpack.from_dlpack(downsampled_points.positions.to_dlpack())
            else:   # point cloud upsampling
                if num_samples/2 < pcd.shape[0]:
                    # pcd 개수가 num_samples/2보다 크면, 부족한 개수만큼 pcd를 랜덤하게 샘플링하여 추가한다.
                    num_of_add_points = num_samples - pcd.shape[0]
                    permuted_indices = torch.randperm(pcd.shape[0], device=self.device)
                    random_indices = permuted_indices[:num_of_add_points]
                    selected_points = pcd[random_indices]
                    torch_tensor_points = torch.cat((pcd, selected_points), dim=0)
                else:
                    # pcd 개수가 num_samples/2보다 작으면, pcd를 반복하여 num_samples개수에 맞춘다.
                    # Calculate the number of iterations to make 100 points
                    repeat_count = (num_samples // pcd.shape[0]) + (1 if num_samples % pcd.shape[0] else 0)
                    repeated_pcd = pcd.repeat(repeat_count, 1)
                    torch_tensor_points = repeated_pcd[:num_samples, :]
            '''
            pcd_np = pcd_pos.detach().cpu().numpy()
            
            
            print(f'pcd shape: {pcd_np.shape}')
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pcd_np)
            o3d.visualization.draw_geometries([point_cloud],
                                                window_name=f'point cloud semantic {idx}')
            '''
            if normalize:
                pcd_mean = torch.mean(pcd, axis=0)
                pcd_mean = torch.unsqueeze(pcd_mean, dim=0)
                pcd_mean_xyz = pcd_mean.repeat(torch_tensor_points.shape[0], 1)
                point_cloud = torch_tensor_points - pcd_mean_xyz    # normalize
            else:
                point_cloud = torch_tensor_points

            # normalized_pcd = normalized_pcd.unsqueeze(0)
            if idx == 0:
                point_clouds = point_cloud
            if idx != 0:
                point_clouds = torch.concat((point_clouds, point_cloud), axis=0)
        
        # normalized_pcds = normalized_pcds.to(self.device)
        return point_clouds

    
    def _visualize_pointcloud(self, idx, env_pos, pcd_data: dict) -> None:
        '''
        try:
            pcd_data = data[annotator]
            env_pos = self.env_pos
        except:
            print('pcd_data is not defined')
            print('Do you mean data[annotator]?')
        
            
        pcd_np = pcd_pos.detach().cpu().numpy()
        pcd_semantic = pcd_semantic.detach().cpu().numpy()
        env_pos = self.env_pos
        idx = 0
        '''

        pcd_np = pcd_data['data']    # get point cloud data as numpy array
        pcd_normal = pcd_data['info']['pointNormals']
        pcd_semantic = pcd_data['info']['pointSemantic']
        
        # lidar_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0.0, 0.0, 0.0]))
        camera_position = np.array(self.camera_positions[idx])
        lidar_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=camera_position)
        
        # base pose is equal to [0, 0, 0]
        org_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array(env_pos[idx]))

        entire_pcd = o3d.geometry.PointCloud()
        entire_pcd.points = o3d.utility.Vector3dVector(pcd_np)
        o3d.visualization.draw_geometries(
                                        #   [entire_pcd, lidar_coord],
                                        #   [entire_pcd, lidar_coord, org_coord],
                                          [entire_pcd, org_coord],
                                        #   [entire_pcd],
                                          window_name=f'entire point cloud, env_{idx}')



        '''
        # TODO: semantic index 순서가 한번씩 뒤바뀌는듯.... 확인 필요
        ### 전체 semantic point cloud를 따로visualize ###
        try:
            pcd_data = data[annotator]
        except:
            print('pcd_data is not defined')
        pcd_np = pcd_data['data']    # get point cloud data as numpy array
        pcd_normal = pcd_data['info']['pointNormals']
        pcd_semantic = pcd_data['info']['pointSemantic']
        print(np.unique(pcd_semantic))
        semantics = np.unique(pcd_semantic)
        for idx in range(semantics.shape[0]):
            print(f'idx: {idx}, semantic: {semantics[idx]}')
            index = np.unique(pcd_semantic)[idx]
            pcd_idx = np.where(pcd_semantic==index)[0]
            pcd = pcd_np[pcd_idx]
            print(f'pcd shape: {pcd.shape}')
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pcd)
            o3d.visualization.draw_geometries([point_cloud],
                                                window_name=f'point cloud semantic {idx}')
            
        # robot
        index_0 = np.unique(pcd_semantic)[0]
        pcd_idx = np.where(pcd_semantic==index_0)[0]
        pcd = pcd_np[pcd_idx]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)
        o3d.visualization.draw_geometries([point_cloud],
                                          window_name='point cloud semantic 0')
        
        # tool
        index_1 = np.unique(pcd_semantic)[1]
        pcd_idx = np.where(pcd_semantic==index_1)[0]
        pcd = pcd_np[pcd_idx]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)
        o3d.visualization.draw_geometries([point_cloud],
                                          window_name='point cloud semantic 1')
        
        # obj
        index_2 = np.unique(pcd_semantic)[2]
        pcd_idx = np.where(pcd_semantic==index_2)[0]
        pcd = pcd_np[pcd_idx]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)
        o3d.visualization.draw_geometries([point_cloud],
                                          window_name='point cloud semantic 2')
        '''

# WriterRegistry.register(PointcloudWriter)
rep.WriterRegistry.register(PointcloudWriter)