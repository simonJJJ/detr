import torch
import torch.nn.functional as F
from modules.grid_attention import ga_weight, ga_map
from modules.grid_align_attention import ga_align_weight, ga_align_map


def ga_weight_pytorch(query, key):
    unfold_op = torch.nn.Unfold(kernel_size=(2, 3), stride=(2, 3))  # (b, c * 4, 100)
    key_unfold = unfold_op(key).view(1, 50, 6, 100)
    query = query.view(1, 50, 100)[:, :, None, :]
    weight = torch.sum(torch.mul(query, key_unfold), dim=1).view(1, 6, 10, 10)
    return weight


def ga_map_pytorch(weight, value):
    unfold_op = torch.nn.Unfold(kernel_size=(2, 3), stride=(2, 3))  # (b, c * 4, 100)
    value_unfold = unfold_op(value).view(1, 50, 6, 100)
    weight = weight.view(1, 6, 100)[:, None, :, :]
    out = torch.sum(torch.mul(value_unfold, weight), dim=2).view(1, 50, 10, 10)
    return out


torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=200, profile="full")
device = torch.device("cuda")
obj_query = torch.randn(size=(1, 50, 10, 10), dtype=torch.double, device=device, requires_grad=True)
feature_map = torch.randn(size=(1, 50, 20, 30), dtype=torch.double, device=device, requires_grad=True)
value = torch.randn(size=(1, 50, 20, 30), dtype=torch.double, device=device, requires_grad=True)

output = ga_weight(obj_query, feature_map)
output_align = ga_align_weight(obj_query, feature_map)
output_pytorch = ga_weight_pytorch(obj_query, feature_map)
assert (output - output_pytorch).abs().max() < 1e-9
assert (output_align - output_pytorch).abs().max() < 1e-9

q_grad1 = torch.autograd.grad(output.mean(), obj_query, retain_graph=True)[0]
q_grad_align = torch.autograd.grad(output_align.mean(), obj_query, retain_graph=True)[0]
q_grad2 = torch.autograd.grad(output_pytorch.mean(), obj_query, retain_graph=True)[0]
assert (q_grad1 - q_grad2).abs().max() < 1e-9
assert (q_grad_align - q_grad2).abs().max() < 1e-9

k_grad1 = torch.autograd.grad(output.mean(), feature_map, retain_graph=True)[0]
k_grad_align = torch.autograd.grad(output_align.mean(), feature_map, retain_graph=True)[0]
k_grad2 = torch.autograd.grad(output_pytorch.mean(), feature_map, retain_graph=True)[0]
assert (k_grad1 - k_grad2).abs().max() < 1e-9
assert (k_grad_align - k_grad2).abs().max() < 1e-9

weight = F.softmax(output, dim=1)
weight_align = F.softmax(output_align, dim=1)
weight_pytorch = F.softmax(output_pytorch, dim=1)

output2 = ga_map(weight, value)
output2_align = ga_align_map(weight_align, value)
output2_pytorch = ga_map_pytorch(weight_pytorch, value)
assert (output2 - output2_pytorch).abs().max() < 1e-9
assert (output2_align - output2_pytorch).abs().max() < 1e-9

w_grad1 = torch.autograd.grad(output2.mean(), weight, retain_graph=True)[0]
w_grad_align = torch.autograd.grad(output2_align.mean(), weight_align, retain_graph=True)[0]
w_grad2 = torch.autograd.grad(output2_pytorch.mean(), weight_pytorch, retain_graph=True)[0]
assert (w_grad1 - w_grad2).abs().max() < 1e-9
assert (w_grad_align - w_grad2).abs().max() < 1e-9

v_grad1 = torch.autograd.grad(output2.mean(), value, retain_graph=True)[0]
v_grad_align = torch.autograd.grad(output2_align.mean(), value, retain_graph=True)[0]
v_grad2 = torch.autograd.grad(output2_pytorch.mean(), value, retain_graph=True)[0]
assert (v_grad1 - v_grad2).abs().max() < 1e-9
assert (v_grad_align - v_grad2).abs().max() < 1e-9

print("test case passed !")
