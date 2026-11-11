import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.bank import ADPM
from networks.module import AttributeFusionPipeline, MVSC
from networks.pretrained_loader import load_hf_convnext_tiny_encoder


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * (x - u) / torch.sqrt(s + self.eps) + self.bias[:, None, None]


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = self.gamma * x if self.gamma is not None else x
        return shortcut + self.drop_path(x.permute(0, 3, 1, 2))


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.conv(self.norm(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class EncoderFusionStage(nn.Module):
    def __init__(self, stage_key, in_channels, img_dim, depth, drop_path_rates, text_dim, feature_hw, attribute_keys):
        super().__init__()
        self.stage_key = stage_key
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, img_dim, kernel_size=4, stride=4),
                LayerNorm(img_dim, eps=1e-6, data_format="channels_first"),
            )
            if stage_key == "x0"
            else DownsampleLayer(in_channels, img_dim, kernel_size=2, stride=2)
        )
        self.stage = nn.Sequential(*[ConvNeXtBlock(img_dim, drop_path_rates[i]) for i in range(depth)])
        self.fusion_pipeline = AttributeFusionPipeline(
            img_dim=img_dim,
            text_dim=text_dim,
            stage_key=stage_key,
            feature_hw=feature_hw,
            attribute_keys=attribute_keys,
        )

    def forward_backbone(self, x):
        return self.stage(self.downsample(x))

    def forward(self, x, current_text_dict, text_compensator, query_bank_fn):
        x = self.forward_backbone(x)
        current_text_dict = text_compensator.forward_stage(self.stage_key, current_text_dict, x)
        current_global_text, bank_feats_dict = query_bank_fn(current_text_dict)
        return self.fusion_pipeline(x, current_global_text, bank_feats_dict), current_text_dict


class DAPSeg(nn.Module):
    def __init__(
        self,
        in_chans=1,
        num_classes=2,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.1,
        input_size=(224, 224),
        attribute_keys=None,
    ):
        super().__init__()
        self.attribute_keys = None if attribute_keys is None else tuple(attribute_keys)
        if not self.attribute_keys:
            raise ValueError("attribute_keys must be provided to DAPSeg.")

        self.input_size = (int(input_size[0]), int(input_size[1]))
        if self.input_size[0] % 32 != 0 or self.input_size[1] % 32 != 0:
            raise ValueError(f"input_size must be divisible by 32, got {self.input_size}.")

        stage_keys = ("x0", "x1", "x2", "x3")
        stage_hw = {key: (self.input_size[0] // (4 << i), self.input_size[1] // (4 << i)) for i, key in enumerate(stage_keys)}
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        stage_drop_rates = [dp_rates[sum(depths[:i]):sum(depths[:i + 1])] for i in range(len(depths))]

        self.text_compensator = MVSC(
            text_dim=dims[3],
            stage_feature_dims=dict(zip(stage_keys, dims)),
            enabled_stage_keys=stage_keys,
        )
        self.fusion_stage_keys = stage_keys
        self.encoder_stages = nn.ModuleDict({
            key: EncoderFusionStage(
                key,
                in_chans if i == 0 else dims[i - 1],
                dims[i],
                depths[i],
                stage_drop_rates[i],
                dims[3],
                stage_hw[key],
                self.attribute_keys,
            )
            for i, key in enumerate(stage_keys)
        })

        self.up3 = UpBlock(dims[3], dims[2], dims[2])
        self.up2 = UpBlock(dims[2], dims[1], dims[1])
        self.up1 = UpBlock(dims[1], dims[0], dims[0])
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(dims[0], dims[0] // 2, kernel_size=2, stride=2),
            nn.GroupNorm(8, dims[0] // 2),
            nn.GELU(),
            nn.ConvTranspose2d(dims[0] // 2, dims[0] // 4, kernel_size=2, stride=2),
            nn.GroupNorm(8, dims[0] // 4),
            nn.GELU(),
        )
        self.seg_head = nn.Conv2d(dims[0] // 4, num_classes, kernel_size=1)
        self.prototype_bank = None
        self._pending_bank_update = None

        self.apply(self._init_weights)
        load_hf_convnext_tiny_encoder(self)

    def _init_prototype_bank(self):
        self.prototype_bank = ADPM(
            attribute_keys=tuple(self.attribute_keys) + ("global",),
            eps=0.01,
            pending_eps=0.05,
            min_samples=2,
            momentum=0.99,
            sim_threshold=0.90,
            max_clusters=30,
            device="cuda",
        )

    def init_bank_from_dataloader(self, dataloader):
        self._init_prototype_bank()
        print(">>> Initializing prototype bank from dataloader.")
        self.eval()

        def extract_encoder_features(images):
            images = images.to(next(self.parameters()).device)
            feats = {}
            x = images
            for stage_key, stage in self.encoder_stages.items():
                x = stage.forward_backbone(x)
                feats[stage_key] = x
            return feats

        self.prototype_bank.build_from_dataloader(dataloader, feature_extractor=extract_encoder_features)
        self.train()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def _forward_encoder_impl(self, x, text_dict=None):
        current_text_dict = text_dict
        self.text_compensator.begin_forward(reference_tensor=x)
        feats = []
        for stage in self.encoder_stages.values():
            x, current_text_dict = stage(x, current_text_dict, self.text_compensator, self._query_bank)
            feats.append(x)
        return (*feats, current_text_dict)

    def forward_decoder(self, features, out_size=None, return_decoder_features=False):
        x0, x1, x2, x3 = features
        d2 = self.up3(x3, x2)
        d1 = self.up2(d2, x1)
        d0 = self.up1(d1, x0)
        logits = self.seg_head(self.final_up(d0))
        if return_decoder_features:
            return logits, {"x0": d0, "x1": d1, "x2": d2, "x3": x3}
        return logits

    def _query_bank(self, text_dict):
        bank_feats_dict = self.prototype_bank.query_all_features(text_dict)
        return text_dict["global"], bank_feats_dict

    def _update_bank_with_fused_features(self, text_dict, fused_features):
        if self.training:
            detached_text = {k: v.detach() for k, v in text_dict.items()}
            detached_features = {k: v.detach() for k, v in fused_features.items()}
            self._pending_bank_update = (detached_text, detached_features)

    @torch.no_grad()
    def commit_pending_bank_update(self, sample_ids=None):
        detached_text, fused_features = self._pending_bank_update
        self.prototype_bank.update_bank(detached_text, fused_features, sample_ids=sample_ids)
        self._pending_bank_update = None

    def forward(self, x, text_dict=None):
        text_dict = {k: F.layer_norm(v, (v.shape[-1],)) for k, v in text_dict.items()}
        x0, x1, x2, x3, text_dict = self._forward_encoder_impl(x, text_dict=text_dict)
        outputs, decoder_features = self.forward_decoder((x0, x1, x2, x3), x.shape[-2:], return_decoder_features=True)
        self._update_bank_with_fused_features(text_dict, decoder_features)
        return outputs


if __name__ == "__main__":
    model = DAPSeg(in_chans=3, num_classes=2, attribute_keys=("attr_1",))
    x = torch.randn(4, 3, 224, 224)
    text = {"attr_1": torch.randn(4, 32, 768), "global": torch.randn(4, 32, 768)}
    y = model(x, text)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
