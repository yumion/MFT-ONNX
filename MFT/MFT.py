from types import SimpleNamespace

import einops
import numpy as np
import torch
import torch.nn as nn

from MFT.results import FlowOUTrackingResult


class MFT(nn.Module):
    def __init__(
        self,
        flower: nn.Module,
        start_frame_i: int = 0,
        time_direction: int = 1,
        occlusion_threshold: float = 0.02,
        deltas: list[int] = [np.inf, 1, 2, 4, 8, 16, 32],
        device: str = "cuda",
    ):
        """MFTクラスの初期化

        Args:
            checkpoint_path: RAFTの学習済みモデルのパス
            start_frame_i: 初期フレームのインデックス
            time_direction: フレームの進む方向 (+1: 前方, -1: 後方)
            occlusion_threshold: オクルージョンの閾値
            deltas: フレーム間の間隔を表すリスト
            device: デバイス (CPU/GPU)
        """
        super().__init__()
        self.flower = flower
        assert time_direction in [+1, -1]
        self.time_direction = time_direction
        self.device = device
        self.occlusion_threshold = occlusion_threshold
        self.deltas = deltas
        self.start_frame_i = start_frame_i
        self.current_frame_i = start_frame_i

    def forward(self, img: torch.Tensor, **kwargs):
        if self.current_frame_i == self.start_frame_i:
            return self.init(img, self.start_frame_i, **kwargs)
        return self.track(img, **kwargs)

    def init(self, img, start_frame_i=0, **kwargs):
        """初期化処理

        Args:
            img: 初期フレーム (torch.Tensor, 形状: [B, C, H, W])
            start_frame_i: 初期フレームのインデックス
        Returns:
            meta: 初期フレームに関する結果 (SimpleNamespace)
        """
        # フレームの高さと幅を記録
        self.img_H, self.img_W = img.shape[2:]

        # メモリに初期フレームを保存
        self.memory = {
            self.start_frame_i: {
                "img": img,
                "result": FlowOUTrackingResult.identity(
                    (self.img_H, self.img_W), device=self.device
                ),
            }
        }

        # 初期化結果を返す
        meta = SimpleNamespace()
        meta.result = self.memory[self.start_frame_i]["result"].clone().cpu()
        self.current_frame_i += self.time_direction  # 現在のフレームのインデックスを更新
        return meta

    def track(self, input_img, **kwargs):
        """1フレームを追跡

        Args:
            input_img: 現在のフレーム (torch.Tensor, 形状: [B, C, H, W])
        Returns:
            meta: 現在のフレームに関する追跡結果
        """
        meta = SimpleNamespace()

        # 複数の`delta`（フレーム間の距離）に基づくOptical Flowを計算
        delta_results = {}
        already_used_left_ids = []

        for delta in self.deltas:
            # フレームペア（left_id, right_id）を計算
            left_id = self.current_frame_i - delta * self.time_direction

            # 初期フレームを超える場合の処理
            if self.is_before_start(left_id):
                if np.isinf(delta):  # 無限距離のデルタの場合は初期フレームを使用
                    left_id = self.start_frame_i
                else:
                    continue
            left_id = int(left_id)

            # 同じペアの計算をスキップ
            if left_id in already_used_left_ids:
                continue

            left_img = self.memory[left_id]["img"]
            right_img = input_img
            template_to_left = self.memory[left_id]["result"]

            # Optical Flowの計算
            flow_left_to_right, occlusions, sigmas = self.flower(left_img, right_img)
            left_to_right = FlowOUTrackingResult(flow_left_to_right[0], occlusions[0], sigmas[0])

            # 結果のチェーン処理
            delta_results[delta] = chain_results(template_to_left, left_to_right)
            already_used_left_ids.append(left_id)

        used_deltas = sorted(
            list(delta_results.keys()), key=lambda delta: 0 if np.isinf(delta) else delta
        )
        all_results = [delta_results[delta] for delta in used_deltas]
        all_flows = torch.stack(
            [result.flow for result in all_results], dim=0
        )  # (N_delta, xy, H, W)
        all_sigmas = torch.stack(
            [result.sigma for result in all_results], dim=0
        )  # (N_delta, 1, H, W)
        all_occlusions = torch.stack(
            [result.occlusion for result in all_results], dim=0
        )  # (N_delta, 1, H, W)

        scores = -all_sigmas
        scores[all_occlusions > self.occlusion_threshold] = -float("inf")

        best = scores.max(dim=0, keepdim=True)
        selected_delta_i = best.indices  # (1, 1, H, W)

        best_flow = all_flows.gather(
            dim=0,
            index=einops.repeat(
                selected_delta_i,
                "N_delta 1 H W -> N_delta xy H W",
                xy=2,
                H=self.img_H,
                W=self.img_W,
            ),
        )
        best_occlusions = all_occlusions.gather(dim=0, index=selected_delta_i)
        best_sigmas = all_sigmas.gather(dim=0, index=selected_delta_i)
        selected_flow, selected_occlusion, selected_sigmas = (
            best_flow,
            best_occlusions,
            best_sigmas,
        )

        selected_flow = einops.rearrange(
            selected_flow, "1 xy H W -> xy H W", xy=2, H=self.img_H, W=self.img_W
        )
        selected_occlusion = einops.rearrange(
            selected_occlusion, "1 1 H W -> 1 H W", H=self.img_H, W=self.img_W
        )
        selected_sigmas = einops.rearrange(
            selected_sigmas, "1 1 H W -> 1 H W", H=self.img_H, W=self.img_W
        )

        result = FlowOUTrackingResult(selected_flow, selected_occlusion, selected_sigmas)

        # mark flows pointing outside of the current image as occluded
        invalid_mask = einops.rearrange(result.invalid_mask(), "H W -> 1 H W")
        result.occlusion[invalid_mask] = 1

        out_result = result.clone()

        meta.result = out_result
        meta.result.cpu()

        self.memory[self.current_frame_i] = {"img": input_img, "result": result}

        # 不要なメモリをクリーンアップ
        self.cleanup_memory()
        self.current_frame_i += self.time_direction  # 現在のフレームのインデックスを更新
        return meta

    # @profile
    def cleanup_memory(self):
        """不要なメモリエントリを削除"""
        # max delta, ignoring the inf special case
        try:
            max_delta = np.amax(np.array(self.deltas)[np.isfinite(self.deltas)])
        except ValueError:  # only direct flow
            max_delta = 0
        has_direct_flow = np.any(np.isinf(self.deltas))
        memory_frames = list(self.memory.keys())
        for mem_frame_i in memory_frames:
            if mem_frame_i == self.start_frame_i and has_direct_flow:
                continue

            if self.time_direction > 0 and mem_frame_i + max_delta > self.current_frame_i:
                # time direction     ------------>
                # mem_frame_i ........ current_frame_i ........ (mem_frame_i + max_delta)
                # ... will be needed later
                continue

            if self.time_direction < 0 and mem_frame_i - max_delta < self.current_frame_i:
                # time direction     <------------
                # (mem_frame_i - max_delta) ........ current_frame_i .......... mem_frame_i
                # ... will be needed later
                continue

            del self.memory[mem_frame_i]

    def is_before_start(self, frame_i):
        """フレームが初期フレームより前かどうかを判定"""
        if self.time_direction > 0:
            return frame_i < self.start_frame_i
        elif self.time_direction < 0:
            return frame_i > self.start_frame_i


def chain_results(left_result, right_result):
    """チェーン結果を生成"""
    flow = left_result.chain(right_result.flow)
    occlusions = torch.maximum(
        left_result.occlusion, left_result.warp_backward(right_result.occlusion)
    )
    sigmas = torch.sqrt(
        torch.square(left_result.sigma)
        + torch.square(left_result.warp_backward(right_result.sigma))
    )
    return FlowOUTrackingResult(flow, occlusions, sigmas)
