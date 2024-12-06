# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------
"""
Deformable DETR model and criterion classes.
"""
#2024.12.05 这个版本在criterion中实现了EDL损失，在后处理postprocess中也加入了EDL计算，待测试
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .segmentation import sigmoid_focal_loss as seg_sigmoid_focal_loss
from .deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    """
    创建模块的N个深度拷贝
    用于构建多层解码器的参数共享
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, num_classes: int = 81, empty_weight: float = 0.1):
    """
    Focal Loss实现
    用途：
    1. 解决类别不平衡问题
    2. 关注难分类的样本
    3. 降低简单样本的权重
    """
    prob = inputs.sigmoid()
    W = torch.ones(num_classes, dtype=prob.dtype, layout=prob.layout, device=prob.device)
    W[-1] = empty_weight
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=W)
    
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class ProbObjectnessHead(nn.Module):
    """
    概率目标检测头
    功能：
    1. 使用BatchNorm计算特征的归一化分数
    2. 评估预测框包含目标的概率
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten(0,1)
        self.objectness_bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def freeze_prob_model(self):
        """
        冻结概率模型的参数
        使BatchNorm层在推理时不更新其统计信息
        """
        self.objectness_bn.eval()
        
    def forward(self, x):
        """
        前向传播
        计算输入特征的归一化分数并返回其范数的平方
        """
        out=self.flatten(x)
        out=self.objectness_bn(out).unflatten(0, x.shape[:2])
        return out.norm(dim=-1)**2
    
    
class FullProbObjectnessHead(nn.Module):
    """
    完整的概率目标检测头
    功能：
    1.维护目标特征的分布统计信息（均值和协方差）
    2.使用马氏距离度量输入特征与已学习的目标分布之间的差异
    3.在训练过程中动态更新分布参数
    4.提供一种概率化的方式来评估预测框是否包含目标
    """
    def __init__(self, hidden_dim=256, device='cpu'):
        super().__init__()
        # 展平层
        self.flatten = nn.Flatten(0, 1)
        # 动量参数
        self.momentum = 0.1
        # 初始化目标特征的均值向量
        self.obj_mean=nn.Parameter(torch.ones(hidden_dim, device=device), requires_grad=False)
        # 初始化目标特征的协方差矩阵
        self.obj_cov=nn.Parameter(torch.eye(hidden_dim, device=device), requires_grad=False)
        # 初始化目标特征协方差矩阵的逆矩阵
        self.inv_obj_cov=nn.Parameter(torch.eye(hidden_dim, device=device), requires_grad=False)
        # 设备
        self.device=device
        # 隐藏层维度
        self.hidden_dim=hidden_dim
            
    def update_params(self,x):
        """
        EMA更新目标特征的均值和协方差
        使用动量更新策略
        """
        # 展平层
        out=self.flatten(x).detach()
        # 计算目标特征的均值
        obj_mean=out.mean(dim=0)
        # 计算目标特征的协方差矩阵
        obj_cov=torch.cov(out.T)
        # 更新均值和协方差矩阵
        self.obj_mean.data = self.obj_mean*(1-self.momentum) + self.momentum*obj_mean
        self.obj_cov.data = self.obj_cov*(1-self.momentum) + self.momentum*obj_cov
        return
    
    def update_icov(self):
        """
        更新协方差矩阵的逆矩阵
        使用伪逆以确保数值稳定性
        """
        self.inv_obj_cov.data = torch.pinverse(self.obj_cov.detach().cpu(), rcond=1e-6).to(self.device)
        return
        
    def mahalanobis(self, x):
        """
        计算马氏距离
        用于评估输入特征与目标特征分布的偏离程度
        """
        # 展平层
        out=self.flatten(x)
        # 计算输入特征与目标特征均值的差
        delta = out - self.obj_mean
        # 计算马氏距离
        m = (delta * torch.matmul(self.inv_obj_cov, delta.T).T).sum(dim=-1)
        return m.unflatten(0, x.shape[:2])
    
    def set_momentum(self, m):
        """
        设置动量参数
        用于控制均值和协方差的更新速度
        """
        self.momentum=m
        return
    
    def forward(self, x):
        """
        前向传播
        在训练时更新参数，返回马氏距离
        """
        if self.training:
            self.update_params(x)
        return self.mahalanobis(x)

class DeformableDETR(nn.Module):
    """
    可变形DETR模型的主体类
    功能：执行目标检测任务，包括特征提取、目标定位和分类
    """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """
        初始化模型
        参数：
        - backbone: 用于特征提取的骨干网络
        - transformer: 用于特征转换的变换器
        - num_classes: 目标类别数
        - num_queries: 查询数，即最大检测目标数
        - num_feature_levels: 特征层数
        - aux_loss: 是否使用辅助损失
        - with_box_refine: 是否进行边界框细化
        - two_stage: 是否使用两阶段检测
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # 在这里调用概率目标检测头
        self.prob_obj_head = ProbObjectnessHead(hidden_dim)

        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        # 初始化类别嵌入的偏置值
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        # 初始化边界框嵌入的权重和偏置
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # 如果使用两阶段检测，则最后一层class_embed和bbox_embed用于区域提议生成
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.prob_obj_head =  _get_clones(self.prob_obj_head, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.prob_obj_head = nn.ModuleList([self.prob_obj_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """
        前向传播
        输入：NestedTensor，包含图像和掩码
        输出：字典，包含预测的分类logits、边界框和辅助输出
        """
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
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        # 查询嵌入
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
        # 输出类别、坐标和目标性
        outputs_classes = []
        outputs_coords = []
        outputs_objectnesses = []
        # 遍历每个特征层
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # 下面这些模型的预测结果会被传入 SetCriterion 进行损失计算

            # 分类预测
            outputs_class = self.class_embed[lvl](hs[lvl])
            # 目标性预测
            outputs_objectness = self.prob_obj_head[lvl](hs[lvl])
            # 边界框预测
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_objectnesses.append(outputs_objectness)

        # 堆叠输出
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_objectness = torch.stack(outputs_objectnesses)

        # 输出字典(包含类别、坐标和目标性)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_obj':outputs_objectness[-1]} 
        # 如果使用辅助损失，则输出辅助损失
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_objectness)
        # 如果使用两阶段检测，则输出编码器输出
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, objectness):
        """
        设置辅助损失
        用于中间层的输出
        """
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_obj': b, 'pred_boxes': c}
                for a, b, c in zip(outputs_class[:-1], objectness[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """
    损失计算 (训练时)
    目的：用于模型训练，计算梯度
    时机：训练阶段
    输入：模型预测和真实标签
    输出：各种损失值的组合，用于反向传播
    """
    """
    损失函数计算类
    主要步骤：
    1. 使用匈牙利算法计算预测框和真实框的最佳匹配
    2. 基于匹配结果计算各种损失（分类损失、框回归损失等）
    """
   
    def __init__(self, num_classes, matcher, weight_dict, losses, invalid_cls_logits, hidden_dim, focal_alpha=0.25):
        """
        初始化损失计算类
        参数：
        - num_classes: 目标类别数
        - matcher: 用于计算目标和预测匹配的模块
        - weight_dict: 各种损失的权重字典
        - losses: 要应用的损失列表
        - invalid_cls_logits: 无效类别的logits
        - hidden_dim: 隐藏层维度
        - focal_alpha: Focal Loss中的alpha参数
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        # 无效类别logits
        self.invalid_cls_logits = invalid_cls_logits
        # 最小目标性（这里是一个OOD阈值）
        self.min_obj = -hidden_dim*math.log(0.9)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        标签损失计算
        outputs接受上面的预测结果字典 out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_obj':outputs_objectness[-1]}
        特点：
        1. 使用EDL(Evidential Deep Learning)损失
        2. 考虑预测的不确定性
        3. 处理无效类别的逻辑
       
        """
        # 预测的类别logits
        assert 'pred_logits' in outputs
        temp_src_logits = outputs['pred_logits'].clone()
        # 将无效类别的logits设置为负无穷
        temp_src_logits[:,:, self.invalid_cls_logits] = -10e10
        src_logits = temp_src_logits

        # 获取预测的索引
        idx = self._get_src_permutation_idx(indices)
        # 获取目标类别
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # 计算 evidence (使用 softplus 确保非负)
        evidence = F.softplus(src_logits)
        
        # 计算 alpha (evidence + 1)
        alpha = evidence + 1
        
        # 计算 uncertainty
        S = torch.sum(alpha, dim=-1, keepdim=True)
        
        # 转换目标为 one-hot 编码
        target_classes = torch.full(src_logits.shape[:2], self.num_classes-1,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_onehot = F.one_hot(target_classes, num_classes=self.num_classes)
        
        # 计算 EDL loss
        loss = torch.sum(target_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)
        
        losses = {'loss_ce': loss.mean()}
        # 如果启用日志记录，则计算分类错误
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        计算基数误差，即预测的非空框数量的绝对误差
        这实际上不是一个损失，仅用于日志记录
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        边界框损失计算
        包含：
        1. L1回归损失
        2. GIoU损失
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

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        如果你的模型不需要进行实例分割任务，可以忽略或移除这个损失函数
        计算与掩码相关的损失：focal loss和dice loss
        目标字典必须包含键"masks"，其值为[nb_target_boxes, h, w]维度的张量
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # 使用有效掩码来屏蔽由于填充而导致的无效区域
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # 将预测上采样到目标大小
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": seg_sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    
    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        """
        计算目标存在的可能性损失，这里是类原型学习的最大似然估计部分
        """
        assert "pred_obj" in outputs
        # 获取预测的索引
        idx = self._get_src_permutation_idx(indices)
        # 获取目标性预测
        pred_obj = outputs["pred_obj"][idx]
        # 计算目标性损失
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    def _get_src_permutation_idx(self, indices):
        """
        根据索引排列预测
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """
        根据索引排列目标
        """
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """
        获取指定类型的损失
        """
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'obj_likelihood': self.loss_obj_likelihood,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        执行损失计算
        参数：
        - outputs: 模型输出的张量字典
        - targets: 目标字典列表，长度等于批量大小
        """
        # 移除辅助输出和编码器输出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k !='pred_obj'}

        # 获取最后一层输出和目标之间的匹配
        indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点上的目标框平均数量，用于归一化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # 如果有辅助损失，则对每个中间层的输出重复此过程
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # 中间层的掩码损失计算成本太高，忽略它们
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # 仅对最后一层启用日志记录
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # 中间层的掩码损失计算成本太高，忽略它们
                    continue
                kwargs = {}
                if loss == 'labels':
                    # 仅对最后一层启用日志记录
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """
    后处理 (推理时)
    目的：将模型输出转换为实际可用的检测结果
    时机：推理阶段
    输入：模型的原始输出
    输出：处理后的检测结果（边界框、类别、分数等）
    """
    """
    后处理模块
    功能：
    1. 将模型输出转换为COCO API格式
    2. 处理预测框的坐标和分数
    3. 执行非极大值抑制(NMS)
    """
    def __init__(self, invalid_cls_logits, temperature=1, pred_per_im=100):
        super().__init__()
        self.temperature=temperature
        self.invalid_cls_logits=invalid_cls_logits
        self.pred_per_im=pred_per_im

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        执行后处理计算
        参数：
        - outputs: 模型的原始输出
        - target_sizes: 张量，包含批量中每个图像的尺寸
        """
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """        
        out_logits, pred_obj, out_bbox = outputs['pred_logits'], outputs['pred_obj'], outputs['pred_boxes']
        out_logits[:,:, self.invalid_cls_logits] = -10e10

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # 计算目标性概率
        obj_prob = torch.exp(-self.temperature*pred_obj).unsqueeze(-1)

        # 计算EDL狄利克雷参数的不确定性
        evidence = F.softplus(out_logits)
        alpha = evidence + 1  # Dirichlet分布的alpha参数

        edlprob = alpha / alpha.sum(dim=-1, keepdim=True)  # 归一化以获得概率分布

        uncertainty = alpha.shape[-1] / alpha.sum(dim=-1, keepdim=True)  # 计算不确定性

        # 计算类别概率（变成了乘以归一化狄利克雷参数）
        prob = obj_prob*edlprob

        # 获取概率最高的k个值和索引
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.pred_per_im, dim=1)
        scores = topk_values
        # 计算边界框索引
        topk_boxes = topk_indexes // out_logits.shape[2]
        # 计算类别索引
        labels = topk_indexes % out_logits.shape[2]
        # 将边界框坐标从cxcywh格式转换为xyxy格式
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # 根据索引收集边界框
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # 返回结果 包含分数、类别、边界框和不确定性
        results = [{'scores': s, 'labels': l, 'boxes': b ,'uncertainty': u} for s, l, b, u in zip(scores, labels, boxes, uncertainty)]
        return results


class MLP(nn.Module):
    """
    多层感知机实现
    用途：
    1. 特征转换
    2. 边界框回归
    3. 提供非线性变换能力
    Very simple multi-layer perceptron (also called FFN)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """
        前向传播
        通过多层感知机进行特征转换
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
    
class ExemplarSelection(nn.Module):
    """
    样本选择模块
    功能：
    1. 选择代表性样本
    2. 计算每个图像的能量分数
    3. 支持增量学习场景
    """
    def __init__(self, args, num_classes, matcher, invalid_cls_logits, temperature=1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.num_seen_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS
        self.invalid_cls_logits=invalid_cls_logits
        self.temperature=temperature
        print(f'running with exemplar_replay_selection')   
              
            
    def calc_energy_per_image(self, outputs, targets, indices):
        """
        计算每个图像的能量分数
        """
        out_logits, pred_obj = outputs['pred_logits'], outputs['pred_obj']
        out_logits[:, :, self.invalid_cls_logits] = -10e10

        torch.exp(-self.temperature*pred_obj).unsqueeze(-1)
        logit_dist = torch.exp(-self.temperature*pred_obj).unsqueeze(-1)
        prob = logit_dist*out_logits.sigmoid()

        image_sorted_scores = {}
        for i in range(len(targets)):
            image_sorted_scores[''.join([chr(int(c)) for c in targets[i]['org_image_id']])] = {'labels':targets[i]['labels'].cpu().numpy(),"scores": prob[i,indices[i][0],targets[i]['labels']].detach().cpu().numpy()}
        return [image_sorted_scores]

    def forward(self, samples, outputs, targets):
        """
        前向传播
        选择代表性样本
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k !='pred_obj'}
        indices = self.matcher(outputs_without_aux, targets)       
        return self.calc_energy_per_image(outputs, targets, indices)


def build(args):
    """
    模型构建函数
    步骤：
    1. 构建backbone和transformer
    2. 初始化DeformableDETR模型
    3. 设置损失函数和后处理器
    4. 配置mask相关组件（如果需要）
    5. 设置exemplar selection机制
    
    参数：
    - args: 包含模型配置的参数对象
    
    返回：
    - model: 训练好的模型
    - criterion: 损失函数
    - postprocessors: 后处理器
    - exemplar_selection: 样本选择器
    """
    # 类别数量
    num_classes = args.num_classes
    # 无效类别logits（从这里设置的阈值）
    invalid_cls_logits = list(range(args.PREV_INTRODUCED_CLS+args.CUR_INTRODUCED_CLS, num_classes-1))
    print("Invalid class range: " + str(invalid_cls_logits))
    
    device = torch.device(args.device)
    
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    
    # 初始化DeformableDETR模型
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    # 匹配器
    matcher = build_matcher(args)
    # 权重字典
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef, 'loss_obj_ll': args.obj_loss_coef}
    
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # 损失函数
    losses = ['labels', 'boxes', 'cardinality','obj_likelihood']
    if args.masks:
        losses += ["masks"]

    # 损失函数
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, invalid_cls_logits, args.hidden_dim, focal_alpha=args.focal_alpha)
    criterion.to(device)
    # 后处理器
    postprocessors = {'bbox': PostProcess(invalid_cls_logits, temperature=args.obj_temp/args.hidden_dim)}
    # 样本选择器
    exemplar_selection = ExemplarSelection(args, num_classes, matcher, invalid_cls_logits, temperature=args.obj_temp/args.hidden_dim)
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors, exemplar_selection
