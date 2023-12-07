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

# class PointcloudListener:
#     """A Observer/Listener that keeps track of updated data sent by the writer. Is passed in the
#     initialization of a PytorchWriter at which point it is pinged by the writer after any data is
#     passed to the writer."""

#     def __init__(self):
#         self.data = {}

#     def write_data(self, data: dict) -> None:
#         """Updates the existing data in the listener with the new data provided.

#         Args:
#             data (dict): new data retrieved from writer.
#         """

#         self.data.update(data)

#     def get_rgb_data(self) -> Optional[torch.Tensor]:
#         """Returns RGB data as a batched tensor from the current data stored.

#         Returns:
#             images (Optional[torch.Tensor]): images in batched pytorch tensor form
#         """

#         if "pytorch_rgb" in self.data:
#             images = self.data["pytorch_rgb"]
#             images = images[..., :3]
#             images = images.permute(0, 3, 1, 2)
#             return images
#         else:
#             return None

#     def get_pointcloud_data(self) -> Optional[torch.Tensor]:
#         if "pointcloud" in self.data:
#             pcd = self.data["pointcloud_data"]
#             # depth = depth[..., :3]
#             # depth = depth.permute(0, 3, 1, 2)
#             return pcd
#         else:
#             return None


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

    def __init__(self, listener: PointcloudListener, output_dir: str = None, device: str = "cuda"):
        # If output directory is specified, writer will write annotated data to the given directory
        if output_dir:
            self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
            self._backend = self.backend
            self._output_dir = self.backend.output_dir
        else:
            self._output_dir = None
        self._frame_id = 0

        # self.annotators = [AnnotatorRegistry.get_annotator("LdrColor", device="cuda", do_array_copy=False),
        #                    AnnotatorRegistry.get_annotator("pointcloud")]
        self.annotators = [AnnotatorRegistry.get_annotator("pointcloud")]
        self.listener = listener
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

        data_tensors = []
        pcd_data = dict()
        for annotator in data.keys():
            if annotator.startswith("pointcloud"):
                if len(annotator.split('_'))==2:
                    idx = f'env_{0}'
                elif len(annotator.split('_'))==3:
                    idx = f'env_{int(annotator.split("_")[-1])}'
                
                pcd = data[annotator]
                pcd_data[idx] = pcd

                # ### 아래 주석은 point cloud를 tensor로 바꾸고 싶을 때 사용
                # pcd_np = data[annotator]['data']
                # pcd_tensor = torch.from_numpy(pcd_np).unsqueeze(0)
                # data_tensors.append(pcd_tensor)

        # Move all tensors to the same device for concatenation
        # device = "cuda:0" if self.device == "cuda" else self.device
        # data_tensors = [t.to(device) for t in data_tensors]
        # data_tensor = torch.cat(data_tensors, dim=0)
        # return data_tensor
        return pcd_data


# WriterRegistry.register(PointcloudWriter)
rep.WriterRegistry.register(PointcloudWriter)
