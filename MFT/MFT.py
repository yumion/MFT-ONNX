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
        checkpoint_path: str = "checkpoints/raft-things-sintel-kubric-splitted-occlusion-uncertainty-non-occluded-base-sintel.pth",
        tracking_level: int = 1,
        flow_iters: int = 12,
        occlusion_module: str = "separate_with_uncertainty",
        small: bool = False,
        mixed_precision: bool = False,
        occlusion_threshold: float = 0.02,
        deltas: list[int] = [np.inf, 1, 2, 4, 8, 16, 32],
        timers_enabled: bool = False,
        device: str = "cuda",
    ):
        """Create MFT tracker"""
        self.flower = RAFTWrapper(
            checkpoint_path=checkpoint_path,
            occlusion_module=occlusion_module,
            small=small,
            mixed_precision=mixed_precision,
            flow_iters=flow_iters,
        )
        self.device = "cuda"
        self.occlusion_threshold = occlusion_threshold
        self.deltas = deltas
        self.timers_enabled = timers_enabled
        self.cache_delta_infinity = None

    def init(self, img, start_frame_i=0, time_direction=1, flow_cache=None, **kwargs):
        """Initialize MFT on first frame

        args:
          img: opencv image (numpy uint8 HxWxC array with B, G, R channel order)
          start_frame_i: [optional] init frame number (used for caching)
          time_direction: [optional] forward = +1, or backward = -1 (used for caching)
          flow_cache: [optional] MFT.utils.io.FlowCache (for caching OF on GPU, RAM, or SSD)
          kwargs: [unused] - for compatibility with other trackers

        returns:
          meta: initial frame result container, with initial (zero-motion) MFT.results.FlowOUTrackingResult in meta.result
        """
        self.img_H, self.img_W = img.shape[:2]
        self.start_frame_i = start_frame_i
        self.current_frame_i = self.start_frame_i
        assert time_direction in [+1, -1]
        self.time_direction = time_direction
        self.flow_cache = flow_cache

        self.memory = {
            self.start_frame_i: {
                "img": img,
                "result": FlowOUTrackingResult.identity(
                    (self.img_H, self.img_W), device=self.device
                ),
            }
        }

        self.template_img = img.copy()

        meta = SimpleNamespace()
        meta.result = self.memory[self.start_frame_i]["result"].clone().cpu()
        return meta

    def track(self, input_img, debug=False, **kwargs):
        """Track one frame

        args:
          input_img: opencv image (numpy uint8 HxWxC array with B, G, R channel order)
          debug: [optional] enable debug visualizations
          kwargs: [unused] - for compatibility with other trackers

        returns:
          meta: current frame result container, with MFT.results.FlowOUTrackingResult in meta.result
                The meta.result represents the accumulated flow field from the init frame, to the current frame
        """
        meta = SimpleNamespace()
        self.current_frame_i += self.time_direction

        # OF(init, t) candidates using different deltas
        delta_results = {}
        already_used_left_ids = []
        chain_timer = general_time_measurer(
            "chain", cuda_sync=True, start_now=False, active=self.timers_enabled
        )
        for delta in self.deltas:
            # candidates are chained from previous result (init -> t-delta) and flow (t-delta -> t)
            # when tracking backward, the chain consists of previous result (init -> t+delta) and flow(t+delta -> t)
            left_id = self.current_frame_i - delta * self.time_direction
            right_id = self.current_frame_i

            # we must ensure that left_id is not behind the init frame
            if self.is_before_start(left_id):
                if np.isinf(delta):
                    left_id = self.start_frame_i
                else:
                    continue
            left_id = int(left_id)

            # because of this, different deltas can result in the same left_id, right_id combination
            # let's not recompute the same candidate multiple times
            if left_id in already_used_left_ids:
                continue

            left_img = self.memory[left_id]["img"]
            right_img = input_img

            template_to_left = self.memory[left_id]["result"]

            flow_init = None
            use_cache = np.isfinite(delta) or self.cache_delta_infinity
            left_to_right = get_flowou_with_cache(
                self.flower,
                left_img,
                right_img,
                flow_init,
                self.flow_cache,
                left_id,
                right_id,
                read_cache=use_cache,
                write_cache=use_cache,
            )

            chain_timer.start()
            delta_results[delta] = chain_results(template_to_left, left_to_right)
            already_used_left_ids.append(left_id)
            chain_timer.stop()

        chain_timer.report("mean")
        chain_timer.report("sum")

        selection_timer = general_time_measurer(
            "selection", cuda_sync=True, start_now=True, active=self.timers_enabled
        )
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
        selection_timer.report()

        out_result = result.clone()

        meta.result = out_result
        meta.result.cpu()

        self.memory[self.current_frame_i] = {"img": input_img, "result": result}

        self.cleanup_memory()
        return meta

    # @profile
    def cleanup_memory(self):
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
        return (
            (self.time_direction > 0 and frame_i < self.start_frame_i)  # forward
            or (self.time_direction < 0 and frame_i > self.start_frame_i)
        )  # backward


# @profile
def get_flowou_with_cache(
    flower,
    left_img,
    right_img,
    flow_init=None,
    cache=None,
    left_id=None,
    right_id=None,
    read_cache=False,
    write_cache=False,
):
    """Compute flow from left_img to right_img. Possibly with caching.

    args:
        flower: flow wrapper
        left_img: (H, W, 3) BGR np.uint8 image
        right_img: (H, W, 3) BGR np.uint8 image
        flow_init: [optional] (2, H, W) tensor with flow initialisation (caching is disabled when flow_init used)
        cache: [optional] cache object with
        left_id: [optional] frame number of left_img
        right_id: [optional] frame number of right_img
        read_cache: [optional] enable loading from flow cache
        write_cache: [optional] enable writing into flow cache

    returns:
        flowou: FlowOUTrackingResult
    """
    must_compute = not read_cache
    if read_cache and flow_init is None:
        # attempt loading cached flow
        assert left_id is not None
        assert right_id is not None

        try:
            assert cache is not None
            flow_left_to_right, occlusions, sigmas = cache.read(left_id, right_id)
            assert flow_left_to_right is not None
        except Exception:
            must_compute = True

    if must_compute:  # read_cache == False, flow not cached yet, or some cache read error
        # print(f'computing flow {left_id}->{right_id}')
        flow_left_to_right, extra = flower.compute_flow(
            left_img, right_img, mode="flow", init_flow=flow_init
        )
        occlusions, sigmas = extra["occlusion"], extra["sigma"]

    if (cache is not None) and write_cache and must_compute and (flow_init is None):
        cache.write(left_id, right_id, flow_left_to_right, occlusions, sigmas)
    flowou = FlowOUTrackingResult(flow_left_to_right, occlusions, sigmas)
    return flowou


def chain_results(left_result, right_result):
    flow = left_result.chain(right_result.flow)
    occlusions = torch.maximum(
        left_result.occlusion, left_result.warp_backward(right_result.occlusion)
    )
    sigmas = torch.sqrt(
        torch.square(left_result.sigma)
        + torch.square(left_result.warp_backward(right_result.sigma))
    )
    return FlowOUTrackingResult(flow, occlusions, sigmas)
