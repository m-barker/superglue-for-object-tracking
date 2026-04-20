# MODIFIED From: https://github.com/gouthamvgk/SuperGlue_training/blob/main/models/superglue.yp
# Originally based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-2, keepdim=True)
        std = x.std(-2, keepdim=True)
        return torch.reshape(self.a_2, (1, -1, 1)) * (
            (x - mean) / (std + self.eps)
        ) + torch.reshape(self.b_2, (1, -1, 1))


def MLP(channels: list, use_layernorm, do_bn=True):
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if use_layernorm:
                layers.append(LayerNorm(channels[i]))
            elif do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers, use_layernorm=False):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim], use_layernorm=use_layernorm)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy"""

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, use_layernorm=False):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP(
            [feature_dim * 2, feature_dim * 2, feature_dim], use_layernorm=use_layernorm
        )
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, use_layernorm=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AttentionalPropagation(feature_dim, 4, use_layernorm=use_layernorm)
                for _ in range(len(layer_names))
            ]
        )
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == "cross":
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, iters: int):
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    assert m == n
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    norm = -(ms + ns).log() # log(1/(M+N))
    log_mu = norm.expand(b, m) # (B, M) every row is weighted by 1/(M+N)
    log_nu = norm.expand(b, n) # (B, N) every col is weighted by 1/(M+N)

    Z = log_sinkhorn_iterations(scores, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """

    default_config = {
        "descriptor_dim": 256,
        "weights_path": None,
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
        "use_layernorm": False,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config["descriptor_dim"],
            self.config["keypoint_encoder"],
            use_layernorm=self.config["use_layernorm"],
        )

        self.gnn = AttentionalGNN(
            self.config["descriptor_dim"],
            self.config["GNN_layers"],
            use_layernorm=self.config["use_layernorm"],
        )

        self.final_proj = nn.Conv1d(
            self.config["descriptor_dim"],
            self.config["descriptor_dim"],
            kernel_size=1,
            bias=True,
        )

        bin_score = torch.nn.Parameter(
            torch.tensor(
                self.config["bin_value"] if "bin_value" in self.config else 1.0
            )
        )
        self.register_parameter("bin_score", bin_score)

        if self.config["weights_path"]:
            weights = torch.load(self.config["weights_path"], map_location="cpu")
            if ("ema" in weights) and (weights["ema"] is not None):
                load_dict = weights["ema"]
            elif "model" in weights:
                load_dict = weights["model"]
            else:
                load_dict = weights
            self.load_state_dict(load_dict)
            print(
                'Loaded SuperGlue model ("{}" weights)'.format(
                    self.config["weights_path"]
                )
            )

    @torch.no_grad()
    def predict(self, data, convert_to_probs: bool = True) -> torch.Tensor:
        self.eval()
        desc0, desc1 = data["descriptors0"], data["descriptors1"]
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]

        # kpts0 = normalize_keypoints(kpts0, data["image0"].shape)
        # kpts1 = normalize_keypoints(kpts1, data["image1"].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data["scores0"])
        desc1 = desc1 + self.kenc(kpts1, data["scores1"])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        scores = scores / self.config["descriptor_dim"] ** 0.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, iters=self.config["sinkhorn_iterations"]
        ) # (B, M, N)

        if convert_to_probs:
            scores = torch.softmax(scores, dim=-1)
            
        self.train()

        return scores

    def forward_train(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        batch_size = data["batch_size"]
        desc0, desc1 = data["descriptors0"], data["descriptors1"]
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]

        # kpts0 = normalize_keypoints(kpts0, data["image0"].shape)
        # kpts1 = normalize_keypoints(kpts1, data["image1"].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data["scores0"])
        desc1 = desc1 + self.kenc(kpts1, data["scores1"])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        scores = scores / self.config["descriptor_dim"] ** 0.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, iters=self.config["sinkhorn_iterations"]
        ) # (B, M, N)

        gt_indexes = data["matches"]
        loss_pre_components = scores[
            gt_indexes[:, 0], gt_indexes[:, 1], gt_indexes[:, 2]
        ]
        loss_pre_components = torch.clamp(loss_pre_components, min=-100, max=0.0)
        loss_vector = -1 * loss_pre_components

        if "loss_mask" in data:
            # Set loss to 0 for placeholder objects to prevent weight updates
            loss_vector *= data["loss_mask"]
            return loss_vector.sum() / data["loss_mask"].sum()

        # TODO:
        # Compute mean of loss vector over batch dimensions
        # Can see if simple mean works for now
        return loss_vector.mean()

if __name__ == "__main__":
    config = {}
    model = SuperGlue(config)
    data = {
        "descriptors0": torch.rand(2, 256, 100),
        "descriptors1": torch.rand(2, 256, 100),
        "keypoints0": torch.rand(2, 100, 2),
        "keypoints1": torch.rand(2, 100, 2),
        "scores0": torch.rand(2, 100),
        "scores1": torch.rand(2, 100),
        "batch_size": 2,
        "matches": # Identity: shouldbe of shape (200, 3) where each row is (batch_index, index_in_desc0, index_in_desc1)
            torch.tensor([[0, i, i] for i in range(100)] + [[1, i, i] for i in range(100)]),
    }
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(1000):
        opt.zero_grad()
        loss = model.forward_train(data)
        loss.backward()
        opt.step()
        print(f"Iteration {i}, loss: {loss.item()}")
