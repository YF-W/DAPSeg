import os

import torch


DEFAULT_CONVNEXT_TINY_PATH = "pretrained/convnext-tiny-224"
CONVNEXT_TINY_STAGE_DEPTHS = (3, 3, 9, 3)


def _resolve_weight_file(local_path):
    return os.path.join(local_path, "pytorch_model.bin") if os.path.isdir(local_path) else local_path


def _unwrap_state_dict(state_dict):
    if "model" in state_dict:
        return state_dict["model"]
    if "state_dict" in state_dict:
        return state_dict["state_dict"]
    return state_dict


def _map_hf_convnext_tiny_state_dict(state_dict):
    mapped_state_dict = {
        "encoder_stages.x0.downsample.0.weight": state_dict["convnext.embeddings.patch_embeddings.weight"],
        "encoder_stages.x0.downsample.0.bias": state_dict["convnext.embeddings.patch_embeddings.bias"],
        "encoder_stages.x0.downsample.1.weight": state_dict["convnext.embeddings.layernorm.weight"],
        "encoder_stages.x0.downsample.1.bias": state_dict["convnext.embeddings.layernorm.bias"],
    }

    for stage_idx, depth in enumerate(CONVNEXT_TINY_STAGE_DEPTHS):
        stage_key = f"x{stage_idx}"
        if stage_idx > 0:
            downsample_prefix = f"convnext.encoder.stages.{stage_idx}.downsampling_layer"
            mapped_state_dict[f"encoder_stages.{stage_key}.downsample.norm.weight"] = state_dict[f"{downsample_prefix}.0.weight"]
            mapped_state_dict[f"encoder_stages.{stage_key}.downsample.norm.bias"] = state_dict[f"{downsample_prefix}.0.bias"]
            mapped_state_dict[f"encoder_stages.{stage_key}.downsample.conv.weight"] = state_dict[f"{downsample_prefix}.1.weight"]
            mapped_state_dict[f"encoder_stages.{stage_key}.downsample.conv.bias"] = state_dict[f"{downsample_prefix}.1.bias"]

        for block_idx in range(depth):
            hf_prefix = f"convnext.encoder.stages.{stage_idx}.layers.{block_idx}"
            model_prefix = f"encoder_stages.{stage_key}.stage.{block_idx}"
            mapped_state_dict[f"{model_prefix}.dwconv.weight"] = state_dict[f"{hf_prefix}.dwconv.weight"]
            mapped_state_dict[f"{model_prefix}.dwconv.bias"] = state_dict[f"{hf_prefix}.dwconv.bias"]
            mapped_state_dict[f"{model_prefix}.norm.weight"] = state_dict[f"{hf_prefix}.layernorm.weight"]
            mapped_state_dict[f"{model_prefix}.norm.bias"] = state_dict[f"{hf_prefix}.layernorm.bias"]
            mapped_state_dict[f"{model_prefix}.pwconv1.weight"] = state_dict[f"{hf_prefix}.pwconv1.weight"]
            mapped_state_dict[f"{model_prefix}.pwconv1.bias"] = state_dict[f"{hf_prefix}.pwconv1.bias"]
            mapped_state_dict[f"{model_prefix}.pwconv2.weight"] = state_dict[f"{hf_prefix}.pwconv2.weight"]
            mapped_state_dict[f"{model_prefix}.pwconv2.bias"] = state_dict[f"{hf_prefix}.pwconv2.bias"]
            layer_scale_key = f"{hf_prefix}.layer_scale_parameter"
            if layer_scale_key in state_dict:
                mapped_state_dict[f"{model_prefix}.gamma"] = state_dict[layer_scale_key]

    return mapped_state_dict


def load_hf_convnext_tiny_encoder(model, local_path=DEFAULT_CONVNEXT_TINY_PATH):
    weight_file = _resolve_weight_file(local_path)
    print(f">>> Loading local ConvNeXt pretrained weights: {weight_file}")
    state_dict = _unwrap_state_dict(torch.load(weight_file, map_location="cpu"))
    mapped_state_dict = _map_hf_convnext_tiny_state_dict(state_dict)
    model.load_state_dict(mapped_state_dict, strict=False)
    print(f">>> Loaded {len(mapped_state_dict)} encoder tensors from local ConvNeXt weights.")
