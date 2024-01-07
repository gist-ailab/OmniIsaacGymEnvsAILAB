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
                 pcd_normalize: bool = True,
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
        v3d = o3d.utility.Vector3dVector
        o3d_org_point_cloud = o3d.geometry.PointCloud()

        device_num = torch.cuda.current_device()
        device = o3d.core.Device(f"{self.device}:{device_num}")
        o3d_t_org_point_cloud = o3d.t.geometry.PointCloud(device)

        for annotator in data.keys():
            if annotator.startswith("pointcloud"):
                if len(annotator.split('_'))==2:
                    # idx = f'env_{0}'
                    idx = 0 # env_0
                elif len(annotator.split('_'))==3:
                    # idx = f'env_{int(annotator.split("_")[-1])}'
                    idx = int(annotator.split("_")[-1]) # env_n

                if self.visualize_point_cloud:
                    self._visualize_pointcloud(idx, self.env_pos, data[annotator])
                pcd_np = data[annotator]['data']    # get point cloud data as numpy array
                pcd_normal = data[annotator]['info']['pointNormals']
                pcd_semantic = data[annotator]['info']['pointSemantic']

                pcd_pos = torch.from_numpy(data[annotator]['data']).to(self.device)
                pcd_normal = torch.from_numpy(data[annotator]['info']['pointNormals']).to(self.device)
                pcd_semantic = torch.from_numpy(data[annotator]['info']['pointSemantic']).to(self.device)

                
                # sampling point cloud
                o3d_tensor = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pcd_pos))
                o3d_pcd_tensor = o3d.t.geometry.PointCloud(o3d_tensor)
                o3d_downsampled_pcd = o3d_pcd_tensor.farthest_point_down_sample(num_samples)
                downsampled_points = o3d_downsampled_pcd.point
                torch_tensor_points = torch.utils.dlpack.from_dlpack(downsampled_points.positions.to_dlpack())


                if self.pcd_normalize:
                    pcd_pos = self._normalize_pcd(
                                                  pcd_pos,
                                                  pcd_normal,
                                                  pcd_semantic,
                                                  idx)

                # TODO: 여기에서 얻은 pcd에 대해 normal vector를 얻어야 한다.
                for i  in torch.unique(pcd_semantic):
                    pcd_idx = torch.where(pcd_semantic==i)[0]
                    pcd_mask = pcd_pos[pcd_idx]
                    # TODO: 여기에서 pcd mask별 normalize 진행


                o3d_tensor = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pcd_pos))
                o3d_pcd_tensor = o3d.t.geometry.PointCloud(o3d_tensor)
                o3d_downsampled_pcd = o3d_pcd_tensor.farthest_point_down_sample(num_samples)
                downsampled_points = o3d_downsampled_pcd.point
                torch_tensor_points = torch.utils.dlpack.from_dlpack(downsampled_points.positions.to_dlpack())
                # TODO: sampling을 mask별로 진행할 것...


                # sampling point cloud for making same size
                o3d_org_point_cloud.points = v3d(pcd_np)
                o3d_downsampled_pcd = o3d_org_point_cloud.farthest_point_down_sample(num_samples)
                pcd_sampled = np.asarray(o3d_downsampled_pcd.points)

                pcd_tensor = torch.from_numpy(pcd_sampled).unsqueeze(0).to(self.device)
                # pcd_tensor = torch.from_numpy(pcd_sampled).to(self.device)

                # if int(idx.split('_')[-1]) == 0:
                if idx == 0:
                    pcd_tensors = pcd_tensor
                else:
                    pcd_tensors = torch.cat((pcd_tensors, pcd_tensor), dim=0)

        return pcd_tensors
    

    def _sampling_pcd(self,
                      pcd_pos,
                      pcd_normal,
                      pcd_semantic,
                      idx: int) -> np.array:
        semantics = torch.unique(pcd_semantic)
        for idx in range(semantics.shape[0]):
            print(f'idx: {idx}, semantic: {semantics[idx]}')
            index = torch.unique(pcd_semantic)[idx]
            pcd_idx = torch.where(pcd_semantic==index)[0]
            pcd = pcd_pos[pcd_idx]
            pcd_mean = torch.mean(pcd, axis=0)
            pcd_mean = torch.unsqueeze(pcd_mean, dim=0)

            pcd_mean_xyz = pcd_mean.repeat(pcd.shape[0], 1)
            normalized_pcd = pcd - pcd_mean_xyz
            if idx == 0:
                normalized_pcds = normalized_pcd
                pcd_means = pcd_mean
            if idx != 0:
                normalized_pcds = torch.concat((normalized_pcds, normalized_pcd), axis=0)
                pcd_means = torch.concat((pcd_means, pcd_mean), axis=0)

        return normalized_pcd, pcd_means


    
    def _normalize_pcd(self,
                       pcd_pos,
                       pcd_normal,
                       pcd_semantic,
                       pcd_means,
                       idx: int) -> np.array:

        semantics = torch.unique(pcd_semantic)
        for idx in range(semantics.shape[0]):
            # print(f'idx: {idx}, semantic: {semantics[idx]}')
            index = torch.unique(pcd_semantic)[idx]
            pcd_idx = torch.where(pcd_semantic==index)[0]
            pcd = pcd_pos[pcd_idx]
            pcd_mean = torch.mean(pcd, axis=0)
            pcd_mean = torch.unsqueeze(pcd_mean, dim=0)
            pcd_mean_xyz = pcd_mean.repeat(pcd.shape[0], 1)
            normalized_pcd = pcd - pcd_mean_xyz
            if idx == 0:
                normalized_pcds = normalized_pcd
            if idx != 0:
                normalized_pcds = torch.concat((normalized_pcds, normalized_pcd), axis=0)

        return normalized_pcd
    
    def _visualize_pointcloud(self, idx, env_pos, pcd_data: dict) -> None:
        '''
        try:
            pcd_data = data[annotator]
            env_pos = self.env_pos
        except:
            print('pcd_data is not defined')
            print('Do you mean data[annotator]?')
        
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