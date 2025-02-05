from typing import List
import torch
import numpy as np
from scipy import sparse as sp
from utils.bboxes import iou

from utils.utils_scripts import from2dot5dto3d, get_center


def normalize(mx: torch.Tensor) -> torch.Tensor:
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_adjacency_dist(
    bboxes: list[list], device: torch.device, sigma: float = 5, top_k: int = -1
) -> torch.Tensor:
    centers = [[get_center(bbox) for bbox in batch] for batch in bboxes]
    n_bboxes = sum([len(batch) for batch in centers])
    adj = torch.zeros((n_bboxes, n_bboxes))
    i = 0
    for batch in centers:
        if not batch:
            continue
        batch = torch.tensor(batch, device=device)
        current_dist_mat = torch.cdist(batch, batch, p=2)
        adj[i : i + len(batch), i : i + len(batch)] = torch.exp(
            -current_dist_mat / sigma
        )
        i += len(batch)

    if top_k > 0:
        knn_values, knn_indices = torch.topk(adj, top_k, dim=1)
        adj = torch.zeros_like(adj)
        adj.scatter_(1, knn_indices, knn_values)

    adj = normalize(adj)
    adj += torch.eye(n_bboxes)
    return torch.from_numpy(adj).to(device)


def get_adjacency_threshold(
    y_true: torch.Tensor,
    bboxes: List[list],
    bb_scale: tuple[float],
    threshold: float = 1.0,
) -> torch.Tensor:
    device = y_true.device
    centers = [[get_center(bbox) for bbox in batch] for batch in bboxes]
    n_bboxes = sum([len(batch) for batch in centers])
    A = torch.zeros((n_bboxes, n_bboxes))
    i = 0
    for batch in centers:
        if not batch:
            continue
        z = y_true[i : i + len(batch)]
        batch = torch.tensor(batch, device=device)
        points_in_3d = from2dot5dto3d(
            torch.cat([batch, z.unsqueeze(1)], dim=1), scale=bb_scale
        )
        current_dist_mat = torch.cdist(points_in_3d, points_in_3d, p=2)
        A[i : i + len(batch), i : i + len(batch)] = (
            current_dist_mat < threshold
        ).float()
        i += len(batch)

    return A.to(device)


def get_adjacency_threshold_coords(
    y_true: torch.Tensor, coords: torch.Tensor, threshold: float = 1.0
) -> torch.Tensor:
    return (torch.cdist(coords, coords, p=2) < threshold).float().to(y_true.device)


def get_adjacency_iou(bboxes: list[list], device: torch.device) -> torch.Tensor:
    n_bboxes = sum([len(batch) for batch in bboxes])
    adj = torch.zeros((n_bboxes, n_bboxes))
    i = 0
    for batch in bboxes:
        if not batch:
            continue
        for j, bbox_a in enumerate(batch):
            for k, bbox_b in enumerate(batch):
                adj[j + i, k + i] = iou(bbox_a, bbox_b)

        i += len(batch)

    return adj.to(device)


def get_laplace_matrix(adj: torch.Tensor, normalize: bool) -> torch.Tensor:
    D = torch.diag(torch.sum(adj, dim=1))
    L = D - adj
    if normalize:
        D_norm = torch.diag(torch.pow(torch.sum(adj, dim=1), -0.5))
        L = D_norm @ L @ D_norm
    return L
