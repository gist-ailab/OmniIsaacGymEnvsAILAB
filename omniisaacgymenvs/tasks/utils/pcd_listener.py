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

class PointcloudListener:
    """A Observer/Listener that keeps track of updated data sent by the writer. Is passed in the
    initialization of a PytorchWriter at which point it is pinged by the writer after any data is
    passed to the writer."""

    def __init__(self):
        self.data = {}

    def write_data(self, data: dict) -> None:
        """Updates the existing data in the listener with the new data provided.

        Args:
            data (dict): new data retrieved from writer.
        """

        self.data.update(data)

    def get_rgb_data(self) -> Optional[torch.Tensor]:
        """Returns RGB data as a batched tensor from the current data stored.

        Returns:
            images (Optional[torch.Tensor]): images in batched pytorch tensor form
        """

        if "pytorch_rgb" in self.data:
            images = self.data["pytorch_rgb"]
            images = images[..., :3]
            images = images.permute(0, 3, 1, 2)
            return images
        else:
            return None

    def get_pointcloud_data(self) -> Optional[torch.Tensor]:
        if self.data is None:
            return None
        else:
            return self.data