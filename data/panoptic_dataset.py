import os
import io
import glob
import torch
import pickle
import numpy as np
import mediapy as media
import random

############### for only two view ##########
import itertools
############################################

from PIL import Image
from typing import Tuple
from einops import rearrange


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)

class PanopticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        dataset_type="panoptic",
        resize_to=224,
        queried_first=True,
        fast_eval=False,
    ):

        local_random = random.Random()
        local_random.seed(42)
        self.fast_eval = fast_eval
        self.dataset_type = dataset_type
        self.resize_to = resize_to
        self.queried_first = queried_first
        

        all_paths = glob.glob(os.path.join(data_root, '*', 'annotation.pkl'))

        points_dataset = {}
        for pickle_path in all_paths:
            seq = pickle_path.split('/')[-2]
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                points_dataset[seq] = data
            
        self.points_dataset = points_dataset
        self.video_names = list(points_dataset.keys())

        self.view_pairs = list(itertools.combinations(range(8), 2))

        self.samples = []
        for video_name in self.video_names:
            for pair in self.view_pairs:
                self.samples.append((video_name, pair))

    def __getitem__(self, index):
        #############
        # if self.dataset_type in ["davis", "robotap", "panoptic"]:
        #     video_name = self.video_names[index]
        # #############
        # else:
        #     video_name = index

        video_name, view_pair_indices = self.samples[index]
        
        # view_pair_indices는 (view1, view2) 형태의 튜플입니다. (예: (0, 3))
        # numpy 인덱싱을 위해 리스트로 변환합니다.
        view_indices_list = list(view_pair_indices)
        
        video = self.points_dataset[video_name]
        frames = video["video"] # (V, T, H, W, 3)
        
        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])
        
        target_points = self.points_dataset[video_name]["coords"] # (V, N, T, 2)
        target_vis = self.points_dataset[video_name]["visibility"] # (V, N, T)

        ## TODO : refactoring
        frames = frames[..., [2, 1, 0]]

        ##############
        frames = frames[view_indices_list]
        target_points = target_points[view_indices_list]
        target_vis = target_vis[view_indices_list]
        ###########

        V, T, H, W, C = frames.shape
        frames = frames.reshape(-1, H, W, C)
        
        frames = resize_video(frames, (self.resize_to, self.resize_to))
        frames = frames.reshape(V, T, self.resize_to, self.resize_to, C)
        target_points *= np.array([self.resize_to/W, self.resize_to/H])

        ## pseudo
        # frames = frames[:2]
        # target_points = target_points[:2]
        # target_vis = target_vis[:2]

        # (V, S, H, W, C) -> (V, S, C, H, W)
        rgbs = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).float()
        # (V, N, S, 2) -> (V, S, N, 2)
        trajs = torch.from_numpy(target_points).permute(0, 2, 1, 3).float()
        # (V, N, S) -> (V, S, N)
        visibles = torch.from_numpy(target_vis).permute(0, 2, 1).bool()

        # Filtering
        flat_pts = rearrange(trajs, 'v s n d -> n (v s) d').numpy()
        flat_occ = ~rearrange(visibles, 'v s n -> n (v s)').numpy() # True=occluded

        is_ever_visible = np.sum(~flat_occ, axis=1) > 0
        
        valid_flat_pts = flat_pts[is_ever_visible]
        valid_flat_occ = flat_occ[is_ever_visible]
        
        trajs = trajs[:, :, is_ever_visible, :]
        visibles = visibles[:, :, is_ever_visible]

        query_points_list = []
        for i in range(valid_flat_pts.shape[0]):
            first_visible_idx = np.where(valid_flat_occ[i] == 0)[0][0]
            x, y = valid_flat_pts[i, first_visible_idx]
            query_points_list.append(np.array([first_visible_idx, y, x]))
        
        query_points = torch.from_numpy(np.stack(query_points_list, axis=0)).float() if query_points_list else torch.empty(0, 3)
        
        # return CoTrackerData(
        #     video=rgbs,          # (V, S, C, H, W)
        #     trajectory=trajs,    # (V, S, N, 2)
        #     visibility=visibles, # (V, S, N)
        #     seq_name=str(video_name),
        #     query_points=query_points,
        # )
        return {
            "video": rgbs.float(),            # (V,T,C,H,W)
            "trajectory": trajs.float(),                         # (V,T,N,2)
            "visibility": visibles.bool(),                     # (V,T,N)
            "seq_name": str(video_name),                                # str (그대로 둬도 됨)
            "query_points": query_points,
            "valid": None
        }

    def __len__(self):
        return len(self.samples)
