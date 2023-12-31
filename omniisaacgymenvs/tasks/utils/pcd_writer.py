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
import carb
import warp as wp
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omniisaacgymenvs.tasks.utils.pcd_listener import PointcloudListener

import os
import open3d as o3d
import numpy as np
import point_cloud_utils as pcu

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
                 num_observations: int = 100,
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
        self.num_observations = num_observations
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
        num_samples = self.num_observations
        v3d = o3d.utility.Vector3dVector
        o3d_org_point_cloud = o3d.geometry.PointCloud()
        for annotator in data.keys():
            if annotator.startswith("pointcloud"):
                if len(annotator.split('_'))==2:
                    idx = f'env_{0}'
                elif len(annotator.split('_'))==3:
                    idx = f'env_{int(annotator.split("_")[-1])}'

                pcd_np = data[annotator]['data']    # get point cloud data as numpy array

                # sampling point cloud for making same size
                o3d_org_point_cloud.points = v3d(pcd_np)
                o3d_downsampled_pcd = o3d_org_point_cloud.farthest_point_down_sample(num_samples)
                pcd_sampled = np.asarray(o3d_downsampled_pcd.points)

                pcd_tensor = torch.from_numpy(pcd_sampled).unsqueeze(0).to(self.device)

                if int(idx.split('_')[-1]) == 0:
                    pcd_tensors = pcd_tensor
                else:
                    pcd_tensors = torch.cat((pcd_tensors, pcd_tensor), dim=0)

        return pcd_tensors


# WriterRegistry.register(PointcloudWriter)
rep.WriterRegistry.register(PointcloudWriter)
