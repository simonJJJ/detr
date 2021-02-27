import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.cross_cc_attention import cca_map, cca_weight


height = 12
width = 10
channels = 64
num_heads = 2
batch_size = 1
num_query = 8


def get_grid_points(points, width, height):
    """
    Args:
        points: (b, num_query, 2)
    """
    grid_points = torch.zeros(size=(batch_size, num_query, height + width - 1, 2), device=points.device, dtype=points.dtype)
    points_x = points[..., 0] * width
    points_y = points[..., 1] * height
    start_x = points_x - torch.floor(points_x)
    start_y = points_y - torch.floor(points_y)
    for i in range(width):
        grid_points[..., i, 0] = start_x + i
        grid_points[..., i, 1] = points_y
    for b_col in range(points.shape[0]):
        for q_col in range(points.shape[1]):
            for j in range(height):
                if (start_y[b_col, q_col] + j) == points_y[b_col, q_col]:
                    continue
                h_index = j if (start_y[b_col, q_col] + j < points_y[b_col, q_col]) else j - 1
                grid_points[b_col, q_col, h_index + width, 0] = points_x[b_col, q_col]
                grid_points[b_col, q_col, h_index + width, 1] = start_y[b_col, q_col] + j
    grid_points[..., 0] /= width
    grid_points[..., 1] /= height
    return grid_points


torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=200, profile="full")
device = torch.device("cuda")
sampling_points = nn.Linear(channels, 2).to(device)
obj_query = torch.randn(size=(batch_size, num_query, channels), dtype=torch.float32, device=device, requires_grad=True)
feature_map = torch.randn(size=(batch_size, channels, height, width), dtype=torch.float32, device=device, requires_grad=True)
feature_map.requires_grad = True
#print("feature map: ", feature_map)

points = sampling_points(obj_query).sigmoid()  # (b, num_query, 2)
cc_points_input = points.unsqueeze(2).expand(1, num_query, num_heads, 2)
grid_points_out = get_grid_points(points, width, height)  # (b, num_query, height + width - 1, 2)
grid_points_pytorch = 2 * grid_points_out - 1
value_pytorch = F.grid_sample(input=feature_map, grid=grid_points_pytorch, align_corners=False)  # (b, channels, num_query, height + width - 1)

# (b, num_query, num_heads, channels, height + width - 1)
value_pytorch = value_pytorch.transpose(1, 2).view(batch_size, num_query, num_heads, channels // num_heads, height + width - 1)
feature_map = feature_map.view(batch_size, num_heads, channels // num_heads, height, width)
obj_query = obj_query.view(batch_size, num_query, num_heads, channels // num_heads)
print("points: ", points)

# (b, num_query, num_heads, height + width - 1)
output_w_pytorch = torch.matmul(obj_query.unsqueeze(3), value_pytorch).squeeze(3)
output_w = cca_weight(obj_query, feature_map, cc_points_input)
print("attn w: ", output_w)
print("attn w torch: ", output_w_pytorch)
q_grad = torch.autograd.grad(output_w.sum(), obj_query, retain_graph=True)[0]
q_grad_pytorch = torch.autograd.grad(output_w_pytorch.sum(), obj_query, retain_graph=True)[0]
print("q_grad: ", q_grad)
print("q_grad_pytorch: ", q_grad_pytorch)
k_grad = torch.autograd.grad(output_w.sum(), feature_map, retain_graph=True)[0]
#k_grad_pytorch = torch.autograd.grad(output_w_pytorch.sum(), feature_map, retain_graph=True, allow_unused=True)[0]
print("k_grad shape: ", k_grad.shape)
#print("k_grad_pytorch shape: ", k_grad_pytorch.shape)
p_grad = torch.autograd.grad(output_w.sum(), points, retain_graph=True)[0]
p_grad_pytorch = torch.autograd.grad(output_w_pytorch.sum(), points, retain_graph=True)[0]
print("p_grad: ", p_grad)
print("p_grad_pytorch: ", p_grad_pytorch)
attn_weight = F.softmax(output_w, -1)  # (b, num_query, num_head, height+width-1)

cc_map = cca_map(attn_weight, feature_map, cc_points_input)
ccmap_pytorch = torch.matmul(attn_weight.unsqueeze(3), value_pytorch.transpose(3, 4)).squeeze(3).flatten(2)
print("ccmap: ", cc_map)
print("ccmap_pytorch: ", ccmap_pytorch)

weight_grad = torch.autograd.grad(cc_map.sum(), attn_weight, retain_graph=True)[0]
weight_grad_pytorch = torch.autograd.grad(ccmap_pytorch.sum(), attn_weight, retain_graph=True)[0]
print("w_grad: ", weight_grad)
print("w_grad_pytorch: ", weight_grad_pytorch)

p_grad_2 = torch.autograd.grad(cc_map.sum(), points, retain_graph=True)[0]
p_grad_pytorch_2 = torch.autograd.grad(ccmap_pytorch.sum(), points, retain_graph=True)[0]
print("p_grad_2: ", p_grad_2)
print("p_grad_pytorch_2: ", p_grad_pytorch_2)
