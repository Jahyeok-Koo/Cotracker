# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import dataclasses
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional, Dict
from dataclasses import is_dataclass, fields

@dataclass(eq=False)
class CoTrackerData:
    """
    Dataclass for storing video tracks data.
    """

    video: torch.Tensor  # B, V, S, C, H, W
    trajectory: torch.Tensor  # B, V, S, N, 2
    visibility: torch.Tensor  # B, V, S, N
    # optional data
    valid: Optional[torch.Tensor] = None  # B, V, S, N
    segmentation: Optional[torch.Tensor] = None  # B, S, 1, H, W
    seq_name: Optional[str] = None
    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format
    transforms: Optional[Dict[str, Any]] = None
    aug_video: Optional[torch.Tensor] = None


# def collate_fn(batch):
#     """
#     Collate function for video tracks data.
#     """
#     video = torch.stack([b.video for b in batch], dim=0)
#     trajectory = torch.stack([b.trajectory for b in batch], dim=0)
#     visibility = torch.stack([b.visibility for b in batch], dim=0)
#     query_points = segmentation = None
#     if batch[0].query_points is not None:
#         query_points = torch.stack([b.query_points for b in batch], dim=0)
#     if batch[0].segmentation is not None:
#         segmentation = torch.stack([b.segmentation for b in batch], dim=0)
#     seq_name = [b.seq_name for b in batch]

#     return CoTrackerData(
#         video=video,
#         trajectory=trajectory,
#         visibility=visibility,
#         segmentation=segmentation,
#         seq_name=seq_name,
#         query_points=query_points,
    

# TODO #########################################################
# Multi view tracking 
# 1. Our data shape VxNxTx2 (V : view) (And also visibility & query too!)
# 2. Modify Collate function code for view dimension! (Is this needed?)
################################################################

# def collate_fn_train(batch_list):
#     # 실패 샘플이 있으면 걸러내기(선택)
#     batch_list = [b for b in batch_list if b is not None]

#     out = {}
#     for k in batch_list[0].keys():
#         vals = [b[k] for b in batch_list]
#         if torch.is_tensor(vals[0]):
#             out[k] = torch.stack(vals, dim=0)   # -> B, ...
#         else:
#             out[k] = vals                       # e.g., list of strings
#     return out

def collate_fn(batch_list):
    # 실패 샘플 제거
    batch_list = [b for b in batch_list if b is not None]
    if len(batch_list) == 0:
        return None  # DataLoader가 다시 뽑도록 하거나 drop_last=True 권장

    out = {}
    keys = batch_list[0].keys()
    for k in keys:
        vals = [b[k] for b in batch_list]

        # None이 섞여 있으면 그냥 None으로 통일
        if any(v is None for v in vals):
            out[k] = None
            continue

        v0 = vals[0]
        if torch.is_tensor(v0):
            try:
                out[k] = torch.stack(vals, dim=0)
            except Exception:
                # shape이 안 맞으면 리스트로 넘김 (후처리에서 다룸)
                out[k] = vals
        else:
            # 문자열 등은 리스트로
            out[k] = vals
    return out



def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj

