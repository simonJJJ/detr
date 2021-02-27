# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
import math
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .cc_transformer import build_transformer


class CCDETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, cc_tr=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        self.input_proj = nn.Sequential(
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim))
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.cc_tr = cc_tr

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        #for proj in self.input_proj:
        nn.init.xavier_uniform_(self.input_proj[0].weight, gain=1)
        nn.init.constant_(self.input_proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        #num_pred = len(transformer.decoder)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])

    def get_reference_points(self, src_shapes, device, stride=32):
        B, C, H, W = src_shapes
        step_h = H / 10
        step_w = W / 10
        bin_size_h = math.floor(step_h)
        bin_size_w = math.floor(step_w)
        ref_y = (torch.arange(0, H, step=step_h, dtype=torch.float32, device=device) + bin_size_h / 2 - 0.5) * stride
        ref_x = (torch.arange(0, W, step=step_w, dtype=torch.float32, device=device) + bin_size_w / 2 - 0.5) * stride
        ref_y, ref_x = torch.meshgrid(ref_y, ref_x)
        #ref_y = ref_y.reshape(-1)[None, :, None].expand(B, -1, 4).flatten(1) / H  # (B, 10*10)
        #ref_x = ref_x.reshape(-1)[None, :, None].expand(B, -1, 4).flatten(1) / W  # (B, 10*10)
        ref_y = (ref_y.reshape(-1)[None].expand(B, -1) + stride // 2) / (H * stride)  # (B, 10*10)
        ref_x = (ref_x.reshape(-1)[None].expand(B, -1) + stride // 2) / (W * stride)  # (B, 10*10)
        ref = torch.stack((ref_x, ref_y), -1)  # (B, 10*10, 2)
        return ref

    def get_locations(self, feature, stride=32):
        h, w = feature.size()[-2:]
        device = feature.device
        step_h = h / 10
        step_w = w / 10
        bin_size_h = math.floor(step_h)
        bin_size_w = math.floor(step_w)
        shifts_y = (torch.arange(0, h, step=step_h, dtype=torch.float32, device=device) + (bin_size_h / 2 - 0.5)) * stride
        shifts_x = (torch.arange(0, w, step=step_w, dtype=torch.float32, device=device) + (bin_size_w / 2 - 0.5)) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs, sampling_points = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])  # [6, b, num_queries, d_model]
        sampling_points = inverse_sigmoid(sampling_points)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            tmp[..., :2] += sampling_points
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        #outputs_class = self.class_embed(hs)
        #outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.focal_alpha = focal_alpha
        #empty_weight = torch.ones(self.num_classes + 1)
        #empty_weight[-1] = self.eos_coef
        #self.register_buffer('empty_weight', empty_weight)

    def get_grid_weight_target(self, outputs, targets):
        stride = float(32)
        stride_r = stride * 1.5
        grid_weight = outputs["pred_grid_weight"]  # (b, 100, 1)
        grid_weight = grid_weight.squeeze()  # (b, 100)
        locations = outputs["locations"]  # (10*10, 2)
        feat_size = outputs["feat_size"]
        bin_size_h = math.floor(feat_size[0] / 10) * stride
        bin_size_w = math.floor(feat_size[1] / 10) * stride
        loc_xs, loc_ys = locations[:, 0], locations[:, 1]  # (100)
        device = grid_weight.device
        grid_weight_labels = []
        inside_gt_bbox_masks = []

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            target_boxes = targets_per_im["boxes"]  # (num_gt_i, 4) cxcywh
            if target_boxes.numel() == 0:
                grid_weight_labels.append(grid_weight.new_zeros(grid_weight.shape[1]))
                continue
            img_h, img_w = targets_per_im["size"][0], targets_per_im["size"][1]
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
            target_boxes = target_boxes * scale_fct
            center_x = target_boxes[:, 0].clone()  # (num_gt_i,)
            center_y = target_boxes[:, 1].clone()  # (num_gt_i,)
            target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)  # (num_gt_i, 4)

            num_gts = target_boxes.shape[0]
            K = locations.shape[0]  # (100,)
            target_boxes = target_boxes[None].expand(K, num_gts, 4)  # (100, num_gt_i, 4)
            center_x = center_x[None].expand(K, num_gts)  # (100, num_gt_i)
            center_y = center_y[None].expand(K, num_gts)  # (100, num_gt_i)
            center_gt = target_boxes.new_zeros(target_boxes.shape)  # (100, num_gt_i, 4)
            xmin = center_x - stride_r - bin_size_w / 2  # (100, num_gt_i)
            ymin = center_y - stride_r - bin_size_h / 2  # (100, num_gt_i)
            xmax = center_x + stride_r + bin_size_w / 2  # (100, num_gt_i)
            ymax = center_y + stride_r + bin_size_h / 2  # (100, num_gt_i)
            center_gt[:, :, 0] = torch.where(xmin > target_boxes[:, :, 0] - bin_size_w / 2, xmin, target_boxes[:, :, 0] - bin_size_w / 2)
            center_gt[:, :, 1] = torch.where(ymin > target_boxes[:, :, 1] - bin_size_h / 2, ymin, target_boxes[:, :, 1] - bin_size_h / 2)
            center_gt[:, :, 2] = torch.where(xmax > target_boxes[:, :, 2] + bin_size_w / 2, target_boxes[:, :, 2] + bin_size_w / 2, xmax)
            center_gt[:, :, 3] = torch.where(ymax > target_boxes[:, :, 3] + bin_size_h / 2, target_boxes[:, :, 3] + bin_size_h / 2, ymax)

            left = loc_xs[:, None] - center_gt[..., 0]  # (100, num_gt_i)
            right = center_gt[..., 2] - loc_xs[:, None]
            top = loc_ys[:, None] - center_gt[..., 1]
            bottom = center_gt[..., 3] - loc_ys[:, None]
            center_bbox = torch.stack((left, top, right, bottom), -1)  # (100, num_gt_i, 4)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # (100, num_gt_i)
            grid_mask = inside_gt_bbox_mask.sum(1) > 0  # (100, )

            grid_weight_label = torch.zeros(loc_xs.shape[0], dtype=torch.float32, device=device)
            grid_weight_label[grid_mask] = 1
            grid_weight_labels.append(grid_weight_label)
            inside_gt_bbox_masks.append(inside_gt_bbox_mask)

        grid_weight_labels = torch.stack(grid_weight_labels, dim=0)  # (b, 100)
        inside_gt_bbox_masks = torch.cat(inside_gt_bbox_masks, dim=1)  # (100, sigma num_gt_i)
        return grid_weight_labels, inside_gt_bbox_masks

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (b, num_queries, 91)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)  # (b, num_queries)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        #loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        #losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_grid_weight(self, outputs, targets, indices, num_boxes):
        assert "pred_grid_weight" in outputs

        grid_weight = outputs["pred_grid_weight"]  # (b, 100, 1)
        grid_weight = grid_weight.squeeze()  # (b, 100)

        loss_grid_weight = F.binary_cross_entropy(grid_weight, self.grid_weight_target)

        losses = {'loss_grid_w': loss_grid_weight}
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'grid_w': self.loss_grid_weight,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        #self.grid_weight_target, self.inside_gt_bbox_masks= self.get_grid_weight_target(outputs, targets)
        #indices = self.matcher(outputs_without_aux, targets, self.inside_gt_bbox_masks)
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        #prob = F.softmax(out_logits, -1)
        #scores, labels = prob[..., :-1].max(-1)

        prob = out_logits.sigmoid()  # (b, num_queries, 91)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)  # (b, 100)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # convert to [x0, y0, x1, y1] format
        #boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    if args.dataset_file == 'kitti_2d':
        num_classes = 4
    if args.dataset_file == 'voc':
        num_classes = 21
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = CCDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        cc_tr=args.cc_tr,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 2, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    #weight_dict['loss_grid_w'] = 1
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
