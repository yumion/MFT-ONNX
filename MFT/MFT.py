# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import logging
from types import SimpleNamespace

import einops
import numpy as np
import torch
import torch.nn as nn

from MFT.raft import RAFTWrapper
from MFT.results import FlowOUTrackingResult
from MFT.utils.timing import general_time_measurer

logger = logging.getLogger(__name__)


class MFT(nn.Module):
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/raft-things-sintel-kubric.pth",
        flow_iters: int = 12,
        occlusion_module: str = "separate_with_uncertainty",
        small: bool = False,
        mixed_precision: bool = False,
        occlusion_threshold: float = 0.02,
        deltas: list[int] = [np.inf, 1, 2, 4, 8, 16, 32],
        timers_enabled: bool = False,
        device: str = "cuda",
    ):
        """MFTクラスの初期化

        Args:
            checkpoint_path: RAFTの学習済みモデルのパス
            flow_iters: RAFTの反復回数
            occlusion_module: オクルージョンモジュールのタイプ
            small: 小型モデルを使うかどうか
            mixed_precision: 混合精度モードを使うかどうか
            occlusion_threshold: オクルージョンの閾値
            deltas: フレーム間の間隔を表すリスト
            timers_enabled: 処理時間計測を有効化するかどうか
            device: デバイス (CPU/GPU)
        """
        super().__init__()
        # Optical Flowを計算するRAFTモデルを初期化
        self.flower = RAFTWrapper(
            checkpoint_path=checkpoint_path,
            occlusion_module=occlusion_module,
            small=small,
            mixed_precision=mixed_precision,
            flow_iters=flow_iters,
        )
        self.device = device
        self.occlusion_threshold = occlusion_threshold
        self.deltas = deltas
        self.timers_enabled = timers_enabled

    def init(self, img, start_frame_i=0, time_direction=1, flow_cache=None, **kwargs):
        """初期化処理

        Args:
            img: 初期フレーム (torch.Tensor, 形状: [B, C, H, W])
            start_frame_i: 初期フレームのインデックス
            time_direction: フレームの進む方向 (+1: 前方, -1: 後方)
            flow_cache: Optical Flowのキャッシュオブジェクト
        Returns:
            meta: 初期フレームに関する結果 (SimpleNamespace)
        """
        # フレームの高さと幅を記録
        self.img_H, self.img_W = img.shape[2:]
        self.start_frame_i = start_frame_i
        self.current_frame_i = self.start_frame_i
        assert time_direction in [+1, -1]
        self.time_direction = time_direction
        self.flow_cache = flow_cache

        # メモリに初期フレームを保存
        self.memory = {
            self.start_frame_i: {
                "img": img,
                "result": FlowOUTrackingResult.identity(
                    (self.img_H, self.img_W), device=self.device
                ),
            }
        }

        # テンプレート画像として初期フレームを保持
        self.template_img = img.clone()

        # 初期化結果を返す
        meta = SimpleNamespace()
        meta.result = self.memory[self.start_frame_i]["result"].clone().cpu()
        return meta

    def track(self, input_img, **kwargs):
        """1フレームを追跡

        Args:
            input_img: 現在のフレーム (torch.Tensor, 形状: [B, C, H, W])
        Returns:
            meta: 現在のフレームに関する追跡結果
        """
        meta = SimpleNamespace()
        self.current_frame_i += self.time_direction  # 現在のフレームのインデックスを更新

        # 複数の`delta`（フレーム間の距離）に基づくOptical Flowを計算
        delta_results = {}
        already_used_left_ids = []
        chain_timer = general_time_measurer(
            "chain", cuda_sync=True, start_now=False, active=self.timers_enabled
        )
        for delta in self.deltas:
            # フレームペア（left_id, right_id）を計算
            left_id = self.current_frame_i - delta * self.time_direction
            right_id = self.current_frame_i

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
            chain_timer.start()
            delta_results[delta] = chain_results(template_to_left, left_to_right)
            already_used_left_ids.append(left_id)
            chain_timer.stop()

        # ベストなデルタを選択
        scores = -torch.stack([delta_results[delta].sigma for delta in self.deltas])
        scores[
            torch.stack([delta_results[delta].occlusion for delta in self.deltas])
            > self.occlusion_threshold
        ] = -float("inf")
        best_delta_idx = torch.argmax(scores, dim=0)
        best_result = delta_results[self.deltas[best_delta_idx.item()]]

        # 結果を更新
        self.memory[self.current_frame_i] = {"img": input_img, "result": best_result}

        # 不要なメモリをクリーンアップ
        self.cleanup_memory()

        meta.result = best_result.clone().cpu()
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
        return (self.time_direction > 0 and frame_i < self.start_frame_i) or (
            self.time_direction < 0 and frame_i > self.start_frame_i
        )


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
