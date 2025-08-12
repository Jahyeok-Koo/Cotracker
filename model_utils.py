import numpy as np
import torch
import torch.nn.functional as F

from einops import rearrange
from typing import Sequence, Mapping, Optional, Tuple
from timm.models.vision_transformer import resize_pos_embed
from peft import get_peft_model, LoraConfig, TaskType

EPS = 1e-6

def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    r"""Masked mean

    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning

    .. math::
        \text{output} =
        \frac
        {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
        {\epsilon + \sum_{i=1}^N \text{mask}_i}

    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.

    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.

    The interface is similar to `torch.mean()`.

    Args:
        inout (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.

    Returns:
        Tensor: mean tensor.
    """

    mask = mask.expand_as(input)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (EPS + denom)
    return mean

def huber_loss(x, y, delta=1.0):
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).float()
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)

def sequence_loss(
    flow_preds,
    flow_gt,
    valids,
    vis=None,
    gamma=0.8,
    add_huber_loss=False,
    loss_only_for_visible=False,
):
    """Loss function defined over sequence of flow predictions"""
    total_flow_loss = 0.0
    for j in range(len(flow_gt)):
        B, S, N, D = flow_gt[j].shape
        B, S2, N = valids[j].shape
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i]
            if add_huber_loss:
                i_loss = huber_loss(flow_pred, flow_gt[j], delta=6.0)
            else:
                i_loss = (flow_pred - flow_gt[j]).abs()  # B, S, N, 2
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            valid_ = valids[j].clone()
            if loss_only_for_visible:
                valid_ = valid_ * vis[j]
            flow_loss += i_weight * reduce_masked_mean(i_loss, valid_)
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss
    return total_flow_loss / len(flow_gt)

def sequence_BCE_loss(vis_preds, vis_gts):
    total_bce_loss = 0.0
    for j in range(len(vis_preds)):
        n_predictions = len(vis_preds[j])
        bce_loss = 0.0
        for i in range(n_predictions):
            vis_loss = F.binary_cross_entropy_with_logits(vis_preds[j][i], vis_gts[j])
            bce_loss += vis_loss
        bce_loss = bce_loss / n_predictions
        total_bce_loss += bce_loss
    return total_bce_loss / len(vis_preds)


def sequence_prob_loss(
    tracks: torch.Tensor,
    confidence: torch.Tensor,
    target_points: torch.Tensor,
    visibility: torch.Tensor,
    expected_dist_thresh: float = 12.0,
):
    """Loss for classifying if a point is within pixel threshold of its target."""
    # Points with an error larger than 12 pixels are likely to be useless; marking
    # them as occluded will actually improve Jaccard metrics and give
    # qualitatively better results.
    total_logprob_loss = 0.0
    for j in range(len(tracks)):
        n_predictions = len(tracks[j])
        logprob_loss = 0.0
        for i in range(n_predictions):
            err = torch.sum((tracks[j][i].detach() - target_points[j]) ** 2, dim=-1)
            valid = (err <= expected_dist_thresh**2).float()
            logprob = F.binary_cross_entropy_with_logits(confidence[j][i], valid, reduction="none")
            logprob *= visibility[j]
            logprob = torch.mean(logprob, dim=[1, 2])
            logprob_loss += logprob
        logprob_loss = logprob_loss / n_predictions
        total_logprob_loss += logprob_loss
    return total_logprob_loss / len(tracks)

def track_loss(
    batch, output, S
):
    # decompose model output
    _, _, _, train_data = output
    coord_predictions, vis_predictions, confidence_predicitons, valid_mask = train_data

    # decompose batch
    trajs_g = batch['trajectory']
    vis_g = batch['visibility'].float()

    valids = batch['valid']
    if isinstance(valids, list):  
        if all(v is None for v in valids):
            valids = None
        elif torch.is_tensor(valids[0]):
            valids = torch.stack(valids, dim=0).to(vis_g.device)
        else:
            valids = None

    if valids is None:
        valids = torch.ones_like(vis_g, dtype=torch.float32, device=vis_g.device)
    else:
        valids = valids.float()

    B, T, N, _ = trajs_g.shape 

    vis_gts = []
    invis_gts = []
    traj_gts = []
    valids_gts = []

    for ind in range(0, T - S // 2, S // 2):
        vis_gts.append(vis_g[:, ind : ind + S].reshape(B, N, -1))
        invis_gts.append(1 - vis_g[:, ind : ind + S].reshape(B, N, -1))
        traj_gts.append(trajs_g[:, ind : ind + S, :, :2].reshape(B, N, -1, 2))
        
        val = valids[:, ind : ind + S].reshape(B, N, -1)
        
        
        val = val * valid_mask[:, ind : ind + S].reshape(B, N, -1)
        
        valids_gts.append(val)

    seq_loss_visible = sequence_loss(
        coord_predictions,
        traj_gts,
        valids_gts,
        vis=vis_gts,
        gamma=0.8,
        add_huber_loss=True,
        loss_only_for_visible=True,
    )
    seq_loss_invisible = sequence_loss(
        coord_predictions,
        traj_gts,
        valids_gts,
        vis=invis_gts,
        gamma=0.8,
        add_huber_loss=False,
        loss_only_for_visible=True,
    )
    confidence_loss = sequence_prob_loss(
        coord_predictions, confidence_predicitons, traj_gts, vis_gts
    )
    vis_loss = sequence_BCE_loss(vis_predictions, vis_gts)

    loss = 0.05*seq_loss_visible.mean() + 0.01*seq_loss_invisible.mean() + confidence_loss.mean() + vis_loss.mean()

    loss_scalars = {}
    loss_scalars['flow'] = 0.05*seq_loss_visible.mean()
    loss_scalars['inv_flow'] = 0.01*seq_loss_invisible.mean()
    loss_scalars['vis'] = vis_loss.mean()
    loss_scalars['conf'] = confidence_loss.mean()
    loss_scalars['loss'] = loss

    return loss, loss_scalars

def get_queries(trajs_g, vis_g, device):
    B, T, N, D = trajs_g.shape 
    __, first_positive_inds = torch.max(vis_g[:, 0], dim=1)
    N_rand = N // 4
    # inds of visible points in the 1st frame

    nonzero_inds = [
        [torch.nonzero(vis_g[b, :, i]) for i in range(N)] for b in range(B)
    ]

    for b in range(B):
        rand_vis_inds = torch.cat(
            [
                nonzero_row[torch.randint(len(nonzero_row), size=(1,))]
                for nonzero_row in nonzero_inds[b]
            ],
            dim=1,
        )
        first_positive_inds[b] = torch.cat(
            [rand_vis_inds[:, :N_rand], first_positive_inds[b : b + 1, N_rand:]],
            dim=1,
        )

    ind_array_ = torch.arange(T, device=device)
    ind_array_ = ind_array_[None, :, None].repeat(B, 1, N)
    # assert torch.allclose(
    #     vis_g[ind_array_ == first_positive_inds[:, None, :]],
    #     torch.ones(1, device=device),
    # )       only now
    gather = torch.gather(
        trajs_g[:, 0], 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, D)
    )
    xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)

    queries = torch.cat([first_positive_inds[:, :, None], xys[:, :, :2]], dim=2)


    if (
        torch.isnan(queries).any()
        or torch.isnan(trajs_g).any()
        or queries.abs().max() > 1500
    ):
        print("failed_sample")
        print("queries time", queries[..., 0])
        print("queries ", queries[..., 1:])
        queries = torch.ones_like(queries).to(queries.device).float()
        print("new queries", queries)
        return None

    return queries

def convert_grid_coordinates(
    coords: torch.Tensor,
    input_grid_size: Sequence[int],
    output_grid_size: Sequence[int],
    coordinate_format: str = 'xy',
) -> torch.Tensor:
  """Convert grid coordinates to correct format."""
  if isinstance(input_grid_size, tuple):
    input_grid_size = torch.tensor(input_grid_size, device=coords.device)
  if isinstance(output_grid_size, tuple):
    output_grid_size = torch.tensor(output_grid_size, device=coords.device)

  if coordinate_format == 'xy':
    if input_grid_size.shape[0] != 2 or output_grid_size.shape[0] != 2:
      raise ValueError(
          'If coordinate_format is xy, the shapes must be length 2.'
      )
  elif coordinate_format == 'tyx':
    if input_grid_size.shape[0] != 3 or output_grid_size.shape[0] != 3:
      raise ValueError(
          'If coordinate_format is tyx, the shapes must be length 3.'
      )
    if input_grid_size[0] != output_grid_size[0]:
      raise ValueError('converting frame count is not supported.')
  else:
    raise ValueError('Recognized coordinate formats are xy and tyx.')

  position_in_grid = coords
  position_in_grid = position_in_grid * output_grid_size / input_grid_size

  return position_in_grid

def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    metrics = {}
    # Fixed bug is described in:
    # https://github.com/facebookresearch/co-tracker/issues/20
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)

    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError("Unknown query mode " + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics

def eval_batch(
    batch, 
    output, 
    eval_metrics_resolution = (256, 256),
    query_first = False,
):
    video = batch['video'].detach().cpu().numpy() # B V T C H W
    gt_vis = batch['visibility'].float()
    query_points = batch['query_points']
    gt_target_points = batch['trajectory']
    traj, vis, conf, _ = output
    vis = vis * conf

    _, T, N, _ = gt_target_points.shape
    traj = traj[:, :, :N]
    vis = vis[:, :, :N]

    # _, V, T, _, _, _ = video.shape

    ## TODO we need to add supprot grid?
    if video.ndim == 6:
        video = rearrange(video, 'b v t c h w -> b (v t) h w c', v=V, t =T)
        gt_target_points = rearrange(gt_target_points, 'b v t n d -> b (v t) n d', v=V, t=T).permute(0, 2, 1, 3)
        gt_occluded = (
            torch.logical_not(rearrange(gt_vis, 'b v t n -> b (v t) n', v=V, t=T).permute(0, 2, 1))
        )
        pred_traj = rearrange(traj, 'b v t n d -> b (v t) n d', v=V, t=T).permute(0, 2, 1, 3)
        pred_occ = (
            torch.logical_not(rearrange(vis, 'b v t n -> b (v t) n', v=V, t=T).permute(0, 2, 1))
        )

    
    query_points = convert_grid_coordinates(
        query_points,
        (1,) + video.shape[2:4],  # (1, height, width)
        (1,) + eval_metrics_resolution,  # (1, height, width)
        coordinate_format='tyx',
    )
    
    gt_target_points = convert_grid_coordinates(
        gt_target_points,
        video.shape[3:1:-1],  # (width, height)
        eval_metrics_resolution[::-1],  # (width, height)
        coordinate_format='xy',
    )
    

    # tracks = output['tracks']
    tracks = convert_grid_coordinates(
        pred_traj,
        video.shape[3:1:-1],  # (width, height)
        eval_metrics_resolution[::-1],  # (width, height)
        coordinate_format='xy',
    )

    query_mode = 'first' if query_first else 'strided'
    metrics = compute_tapvid_metrics(
        query_points=query_points.detach().cpu().numpy(),
        gt_occluded=gt_occluded.detach().cpu().numpy(),
        gt_tracks=gt_target_points.detach().cpu().numpy(),
        pred_occluded=pred_occ.detach().cpu().numpy(),
        pred_tracks=tracks.detach().cpu().numpy(),
        query_mode=query_mode,
    )

    return metrics

def construct_lora_vggt(
        model,
        lora_path,
        lora_rank,
):

    state_dict = torch.load(lora_path)
    agg_prefix = 'aggregator.'
    agg_state_dict = {
        k[len(agg_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(agg_prefix)
    }

    track_dpt_prefix = 'track_head.feature_extractor.'
    track_dpt_state_dict = {
        k[len(track_dpt_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(track_dpt_prefix)
    }

    old_pos = agg_state_dict['patch_embed.pos_embed'] 
    new_pos = model.fnet.patch_embed.pos_embed 

    resized_pos = resize_pos_embed(
        old_pos,
        new_pos,
        num_prefix_tokens=1
    )

    agg_state_dict['patch_embed.pos_embed'] = resized_pos

    model.fnet.load_state_dict(agg_state_dict, strict=True)
    model.dpt.load_state_dict(track_dpt_state_dict, strict=True)
    

    for param in model.fnet.parameters():
        param.requires_grad = False
    
    model.fnet.patch_embed.pos_embed.requires_grad = True

    target = [
            "global_blocks.4.attn.qkv",
            "global_blocks.4.attn.proj",
            "global_blocks.4.mlp.fc1",
            "global_blocks.4.mlp.fc2",
            "global_blocks.11.attn.qkv",
            "global_blocks.11.attn.proj",
            "global_blocks.11.mlp.fc1",
            "global_blocks.11.mlp.fc2",
            "global_blocks.17.attn.qkv",
            "global_blocks.17.attn.proj",
            "global_blocks.17.mlp.fc1",
            "global_blocks.17.mlp.fc2",
            "global_blocks.23.attn.qkv",
            "global_blocks.23.attn.proj",
            "global_blocks.23.mlp.fc1",
            "global_blocks.23.mlp.fc2",
            "frame_blocks.4.attn.qkv",
            "frame_blocks.4.attn.proj",
            "frame_blocks.4.mlp.fc1",
            "frame_blocks.4.mlp.fc2",
            "frame_blocks.11.attn.qkv",
            "frame_blocks.11.attn.proj",
            "frame_blocks.11.mlp.fc1",
            "frame_blocks.11.mlp.fc2",
            "frame_blocks.17.attn.qkv",
            "frame_blocks.17.attn.proj",
            "frame_blocks.17.mlp.fc1",
            "frame_blocks.17.mlp.fc2",
            "frame_blocks.23.attn.qkv",
            "frame_blocks.23.attn.proj",
            "frame_blocks.23.mlp.fc1",
            "frame_blocks.23.mlp.fc2",  
    ]

    lora_config = LoraConfig(
        r=lora_rank, 
        lora_alpha=32,
        target_modules=target, 
        lora_dropout=0.05,
        bias="none", 
        task_type=TaskType.FEATURE_EXTRACTION 
    )
    model.fnet = get_peft_model(model.fnet, lora_config)

    model.fnet.print_trainable_parameters()


def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)