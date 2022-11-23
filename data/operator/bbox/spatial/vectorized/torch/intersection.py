import torch


def bbox_compute_intersection_vectorized(bbox_a: torch.Tensor, bbox_b: torch.Tensor):
    if bbox_b.get_device() != bbox_a.get_device():
        device = bbox_b.get_device()
        bbox_a = bbox_a.to(device)
    intersection = torch.cat((torch.maximum(bbox_a[..., :2], bbox_b[..., :2]), torch.minimum(bbox_a[..., 2:], bbox_b[..., 2:])), dim=-1)
    intersection[(intersection[..., 2] - intersection[..., 0]) <= 0, ...] = 0
    intersection[(intersection[..., 3] - intersection[..., 1]) <= 0, ...] = 0
    return intersection
