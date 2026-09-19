import torch

def normalize_depth_to_255(depth_map):
    assert (
        depth_map.ndim == 4 and depth_map.size(1) == 1
    ), "depth_map must be (B, 1, H, W)"

    B, _, H, W = depth_map.shape
    
    normalized = torch.zeros_like(depth_map, dtype=torch.float32)

    for b in range(B):
        depth = depth_map[b, 0]  # (H, W)
        valid_mask = depth > 0

        if valid_mask.sum() == 0:
            #* 无效深度维持零图，proposal 后续也为空，不制造假三维点。
            continue

        valid_depth = depth[valid_mask]
        d_min = valid_depth.min()
        d_max = valid_depth.max()

        if d_max == d_min:
            normalized[b, 0][valid_mask] = 255.0
        else:
            normed = (depth[valid_mask] - d_min) / (d_max - d_min) * 255.0
            normalized[b, 0][valid_mask] = normed

    return normalized

def normalize(img, mean, std):
    assert img.ndim == 4, "img must be (B, C, H, W)"

    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=img.device, dtype=img.dtype)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=img.device, dtype=img.dtype)

    # (B,C,H,W) -> (B,C,H,W)
    img = img / 255.0
    
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    img = (img - mean) / std

    return img
