import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from MFT.RAFT.core.raft import RAFT  # noqa: E402
from MFT.RAFT.core.utils.utils import InputPadder  # noqa: E402

logger = logging.getLogger(__name__)


class RAFTWrapper(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        flow_iters: int = 12,
        occlusion_module: str = "separate_with_uncertainty",
        small: bool = False,
        mixed_precision: bool = False,
    ):
        super().__init__()
        device = "cuda"
        self.flow_iters = flow_iters

        model = torch.nn.DataParallel(
            RAFT(
                occlusion_module=occlusion_module,
                small=small,
                mixed_precision=mixed_precision,
            )
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        model = model.module
        model.requires_grad_(False)
        model.to(device)
        model.eval()

        self.model = model

    def forward(self, image1: torch.Tensor, image2: torch.Tensor):
        """
        Args:
            image1 (torch.Tensor): The first image tensor, shape (1, 3, H, W).
            image2 (torch.Tensor): The second image tensor, shape (1, 3, H, W).
        Returns:
            torch.Tensor: Predicted flow tensor, shape (1, 2, H, W).
        """
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        all_predictions = self.model(
            image1,
            image2,
            iters=self.flow_iters,
            test_mode=True,
        )

        flow = padder.unpad(all_predictions["flow"])  # (1, 2, H, W)
        occlusions = padder.unpad(
            all_predictions["occlusion"].softmax(dim=1)[:, 1:2, :, :]
        )  # (1, 1, H, W)
        uncertainty_pred = padder.unpad(all_predictions["uncertainty"])  # (1, 1, H, W)
        sigma = torch.sqrt(torch.exp(uncertainty_pred))

        return flow, occlusions, sigma


def downsample_flow_8(flow, mode="bilinear"):
    """Downsample a (B, xy, H, W) flow tensor to (B, xy, H/8, W/8) (assume divisible)"""
    new_size = (flow.shape[2] // 8, flow.shape[3] // 8)
    return F.interpolate(flow, size=new_size, mode=mode, align_corners=True) / 8
