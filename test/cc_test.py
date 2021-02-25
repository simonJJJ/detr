import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.cross_cc_attention import cca_map, cca_weight


torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=200, profile="full")
device = torch.device("cuda")
sampling_points = nn.Linear(10, 2).to(device)
obj_query = torch.randn(size=(1, 8, 10), dtype=torch.float32, device=device, requires_grad=True)
feature_map = torch.randn(size=(1, 10, 12, 13), dtype=torch.float32, device=device, requires_grad=True)
feature_map = feature_map.view(1, 2, 5, 12, 13)
print("feature map: ", feature_map)
points = sampling_points(obj_query).view(1, 8, 2).sigmoid()
obj_query = obj_query.view(1, 8, 2, 5)
print("points: ", points)
output_w = cca_weight(obj_query, feature_map, points)
attn_weight = F.softmax(output_w, -1)
print("attn weight: ", output_w)

query_grad = torch.autograd.grad(output_w.mean(), obj_query, retain_graph=True)[0]
key_grad = torch.autograd.grad(output_w.mean(), feature_map, retain_graph=True)[0]
points_grad = torch.autograd.grad(output_w.mean(), points, retain_graph=True)[0]
print("query grad shape: ", query_grad.shape)
print("query grad: ", query_grad)
print("key grad shape: ", key_grad.shape)
print("key grad: ", key_grad)
print("points grad shape: ", points_grad.shape)
print("points grad: ", points_grad)

output = cca_map(attn_weight, feature_map, points)
print("cca forward shape: ", output.shape)
print("cca forward: ", output)

value_grad = torch.autograd.grad(output.mean(), feature_map, retain_graph=True)[0]
weight_grad = torch.autograd.grad(output.mean(), attn_weight, retain_graph=True)[0]
points_grad = torch.autograd.grad(output.mean(), points, retain_graph=True)[0]
print("value grad shape: ", value_grad.shape)
print("value grad: ", value_grad)
print("weight grad shape: ", weight_grad.shape)
print("weight grad: ", weight_grad)
print("points grad shape: ", points_grad.shape)
print("points grad: ", points_grad)
