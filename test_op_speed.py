import torch
import time
import torch.nn.functional as F
from modules.cc_attention import ca_weight, ca_map


cuda1 = torch.device('cuda:0')
t_map = torch.randn((2, 256, 512, 512), dtype=torch.float32, device=cuda1)
weight = torch.empty((3 * 512, 512), dtype=torch.float32, device=cuda1)
weight_conv1 = torch.empty((32, 256, 1, 1), dtype=torch.float32, device=cuda1)
weight_conv2 = torch.empty((32, 256, 1, 1), dtype=torch.float32, device=cuda1)
weight_conv3 = torch.empty((256, 256, 1, 1), dtype=torch.float32, device=cuda1)
weight_conv_out1 = torch.empty((256, 256, 3, 3), dtype=torch.float32, device=cuda1)
weight_conv_out2 = torch.empty((256, 256, 3, 3), dtype=torch.float32, device=cuda1)
running_mean = torch.zeros(256, dtype=torch.float32, device=cuda1)
running_var = torch.ones(256, dtype=torch.float32, device=cuda1)


def normal_attn(feat):
    feat.flatten(2).permute(2, 0, 1)
    q, k, v = F.linear(feat, weight).chunk(3, dim=-1)
    q = q.contiguous().view(256, 2, 512 * 512).transpose(0, 1)
    k = k.contiguous().view(256, 2, 512 * 512).transpose(0, 1)
    v = v.contiguous().view(256, 2, 512 * 512).transpose(0, 1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output = torch.bmm(attn_output_weights, v)
    return attn_output


def cc_attn(feat):
    feat = F.relu(F.batch_norm(F.conv2d(feat, weight_conv_out1), running_mean, running_var))
    for _ in range(2):
        q = F.conv2d(feat, weight_conv1)
        k = F.conv2d(feat, weight_conv2)
        v = F.conv2d(feat, weight_conv3)

        energy = ca_weight(q, k)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, v)
    out = F.relu(F.batch_norm(F.conv2d(out, weight_conv_out2), running_mean, running_var))
    return out


tic1 = time.time()
out1 = normal_attn(t_map)
normal_attn_time = time.time() - tic1
print("noraml attention time: ", normal_attn_time)

tic2 = time.time()
out2 = cc_attn(t_map)
cc_attn_time = time.time() - tic2
print("cc attention time: ", cc_attn_time)