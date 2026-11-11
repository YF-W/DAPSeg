import torch
import torch.nn as nn
import torch.nn.functional as F

def _resolve_num_heads(embed_dim, preferred_heads=8):
    for num_heads in range(min(preferred_heads, embed_dim), 0, -1):
        if embed_dim % num_heads == 0:
            return num_heads
    return 1


class ConditionalPositionEncoding2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.proj(x)


class GAEE(nn.Module):
    def __init__(self, img_dim, text_dim=768, stage_key=None, feature_hw=None, num_heads=8, attribute_keys=None):
        super().__init__()
        if len(feature_hw) != 2:
            raise ValueError("feature_hw must be a pair of (height, width).")

        self.img_dim = img_dim
        self.text_dim = text_dim
        self.stage_key = stage_key
        self.attribute_keys = tuple(attribute_keys) if attribute_keys is not None else None
        if not self.attribute_keys:
            raise ValueError("attribute_keys must be provided to GAEE.")
        self.feature_hw = (int(feature_hw[0]), int(feature_hw[1]))
        self.pixel_count = self.feature_hw[0] * self.feature_hw[1]
        self.attr_num_heads = _resolve_num_heads(img_dim, num_heads)
        self.channel_num_heads = _resolve_num_heads(self.pixel_count, num_heads)
        self.channel_head_dim = self.pixel_count // self.channel_num_heads
        self.bank_image_pos = ConditionalPositionEncoding2D(img_dim)

        self.img_global_attn = nn.MultiheadAttention(
            embed_dim=img_dim,
            num_heads=self.attr_num_heads,
            kdim=text_dim,
            vdim=text_dim,
            batch_first=True,
        )
        self.prior_image_proj = nn.Linear(img_dim, img_dim, bias=False)
        self.prior_text_proj = nn.Linear(text_dim, img_dim, bias=False)
        self.q_spatial_proj = nn.Linear(self.pixel_count, self.pixel_count)
        self.k_route_proj = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.v_route_proj = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.q_dynamic_scale = nn.Parameter(torch.tensor(1.0))
        self.k_route_scale = nn.Parameter(torch.tensor(1.0))
        self.v_route_scale = nn.Parameter(torch.tensor(1.0))
        self.attr_out_proj = nn.Linear(self.pixel_count, self.pixel_count)
        in_channels = len(self.attribute_keys) * self.img_dim
        self.concat_proj = nn.Conv2d(in_channels, self.img_dim, kernel_size=1, bias=False)
        self.prior_proj = nn.Conv2d(self.img_dim, self.img_dim, kernel_size=3, padding=1)
        self.prior_norm = nn.BatchNorm2d(self.img_dim)

    def forward(self, x, bank_feats_dict):
        B, C, H, W_img = x.shape
        if (H, W_img) != self.feature_hw:
            raise ValueError(
                f"Stage '{self.stage_key}' expected feature size {self.feature_hw}, got {(H, W_img)}."
            )
        extracted_features_dict = {}

        for attr_key, bank_data in bank_feats_dict.items():
            b_text = bank_data['text']
            if b_text.dim() != 3 or b_text.shape[0] != B or b_text.shape[-1] != self.text_dim:
                raise ValueError(
                    f"Bank text for attribute '{attr_key}' must have shape [B, N, {self.text_dim}], "
                    f"got {tuple(b_text.shape)}."
                )

            b_img_map = bank_data['image'][self.stage_key]
            if b_img_map.shape[-2:] != (H, W_img):
                raise ValueError(
                    f"Bank image feature for attribute '{attr_key}' stage '{self.stage_key}' "
                    f"has size {tuple(b_img_map.shape[-2:])}, expected {(H, W_img)}."
                )
            b_img = self.bank_image_pos(b_img_map).flatten(2).transpose(1, 2).contiguous()

            b_global = bank_data['global']

            global_guided_tokens, _ = self.img_global_attn(
                b_img,
                b_global,
                b_global,
                need_weights=False,
            )
            prior_image = self.prior_image_proj(global_guided_tokens)
            prior_text = self.prior_text_proj(b_text)
            prior_logits = torch.bmm(prior_image, prior_text.transpose(1, 2)) * (C ** -0.5)

            pixel_count = H * W_img
            token_count = b_text.shape[1]
            image_channels = b_img.transpose(1, 2).contiguous()
            q_base = self.q_spatial_proj(image_channels)
            q_dynamic = torch.bmm(torch.bmm(image_channels, prior_logits), prior_logits.transpose(1, 2))
            q = q_base + self.q_dynamic_scale * q_dynamic / max(1, token_count)

            route_maps = prior_logits.transpose(1, 2).contiguous().view(B * token_count, 1, H, W_img)
            route_base = route_maps.flatten(2).squeeze(1).view(B, token_count, pixel_count)
            k_routes = route_base + self.k_route_scale * self.k_route_proj(route_maps).flatten(2).squeeze(1).view(B, token_count, pixel_count)
            v_routes = route_base + self.v_route_scale * self.v_route_proj(route_maps).flatten(2).squeeze(1).view(B, token_count, pixel_count)
            route_scale = max(1, token_count) ** -0.5
            text_channels = b_text.transpose(1, 2).contiguous()
            k = torch.bmm(text_channels, k_routes) * route_scale
            v = torch.bmm(text_channels, v_routes) * route_scale

            q = q.view(B, C, self.channel_num_heads, self.channel_head_dim).transpose(1, 2)
            k = k.view(B, self.text_dim, self.channel_num_heads, self.channel_head_dim).transpose(1, 2)
            v = v.view(B, self.text_dim, self.channel_num_heads, self.channel_head_dim).transpose(1, 2)
            attr_tokens = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
            )
            attr_tokens = attr_tokens.transpose(1, 2).contiguous().view(B, C, pixel_count)
            attr_tokens = self.attr_out_proj(attr_tokens)
            attr_img_feat = attr_tokens.view(B, C, H, W_img)
            
            extracted_features_dict[attr_key] = attr_img_feat

        concatenated_features = torch.cat(
            [extracted_features_dict[attr_key] for attr_key in self.attribute_keys],
            dim=1,
        )
        fused_features = self.concat_proj(concatenated_features)
        return self.prior_norm(self.prior_proj(fused_features))

class GAWF(nn.Module):
    def __init__(self, img_dim, text_dim=768, feature_hw=None, num_heads=8):
        super().__init__()
        if len(feature_hw) != 2:
            raise ValueError("feature_hw must be a pair of (height, width).")

        self.img_dim = img_dim
        self.text_dim = text_dim
        self.num_heads = _resolve_num_heads(img_dim, num_heads)
        self.feature_hw = (int(feature_hw[0]), int(feature_hw[1]))
        self.high_freq_hw = ((self.feature_hw[0] + 1) // 2, (self.feature_hw[1] + 1) // 2)
        self.high_freq_pixel_dim = self.high_freq_hw[0] * self.high_freq_hw[1]
        self.high_freq_num_heads = _resolve_num_heads(self.high_freq_pixel_dim, num_heads)
        self.global_text_proj = nn.Identity() if text_dim == img_dim else nn.Linear(text_dim, img_dim)
        self.align_image_pos = ConditionalPositionEncoding2D(img_dim)
        self.low_freq_pos = ConditionalPositionEncoding2D(img_dim)
        
        self.align_attn = nn.MultiheadAttention(
            embed_dim=img_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.low_freq_attn = nn.MultiheadAttention(
            embed_dim=img_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.high_freq_attn = nn.MultiheadAttention(
            embed_dim=self.high_freq_pixel_dim,
            num_heads=self.high_freq_num_heads,
            batch_first=True,
        )

        self.high_prior_down = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(img_dim, img_dim, kernel_size=1, bias=False),
        )
        self.low_prior_down = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(img_dim, img_dim, kernel_size=1, bias=False),
        )
        self.out_proj = nn.Conv2d(img_dim, img_dim, kernel_size=3, padding=1)

    def _dwt(self, x):
        B, C, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            
        x00 = x[:, :, 0::2, 0::2].contiguous()
        x01 = x[:, :, 0::2, 1::2].contiguous()
        x10 = x[:, :, 1::2, 0::2].contiguous()
        x11 = x[:, :, 1::2, 1::2].contiguous()
        
        ll = (x00 + x01 + x10 + x11) / 2.0
        hl = (x00 - x01 + x10 - x11) / 2.0
        lh = (x00 + x01 - x10 - x11) / 2.0
        hh = (x00 - x01 - x10 + x11) / 2.0
        return ll, hl, lh, hh, pad_h, pad_w

    def _idwt(self, ll, hl, lh, hh, pad_h, pad_w):
        B, C, H2, W2 = ll.shape
        x = torch.zeros(B, C, H2 * 2, W2 * 2, device=ll.device, dtype=ll.dtype)
        x[:, :, 0::2, 0::2] = (ll + hl + lh + hh) / 2.0
        x[:, :, 0::2, 1::2] = (ll - hl + lh - hh) / 2.0
        x[:, :, 1::2, 0::2] = (ll + hl - lh - hh) / 2.0
        x[:, :, 1::2, 1::2] = (ll - hl - lh + hh) / 2.0
        
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :x.shape[2]-pad_h, :x.shape[3]-pad_w].contiguous()
        return x

    def forward(self, x, attribute_prior, current_global_text):
        B, C, H, W = x.shape
        if (H, W) != self.feature_hw:
            raise ValueError(
                f"GAWF expected feature size {self.feature_hw}, got {(H, W)}."
            )
        A = attribute_prior

        G = current_global_text
        G = self.global_text_proj(G)

        q_align = self.align_image_pos(x).flatten(2).transpose(1, 2).contiguous()
        G_dynamic_tokens, _ = self.align_attn(q_align, G, G, need_weights=False)
        G_dynamic = G_dynamic_tokens.transpose(1, 2).contiguous().view(B, C, H, W)

        dot_AG = (A * G_dynamic).sum(dim=1, keepdim=True)         
        dot_GG = (G_dynamic * G_dynamic).sum(dim=1, keepdim=True) + 1e-6 

        A_parallel_G = ((dot_AG / dot_GG) * G_dynamic).contiguous()
        G_perp_A = G_dynamic

        ll, hl, lh, hh, pad_h, pad_w = self._dwt(x)
        _, _, Hd, Wd = ll.shape
        if (Hd, Wd) != self.high_freq_hw:
            raise ValueError(
                f"GAWF expected DWT size {self.high_freq_hw}, got {(Hd, Wd)}."
            )

        if pad_h > 0 or pad_w > 0:
            A_parallel_G = F.pad(A_parallel_G, (0, pad_w, 0, pad_h))
            G_perp_A = F.pad(G_perp_A, (0, pad_w, 0, pad_h))


        A_parallel_G_d = self.high_prior_down(A_parallel_G)
        G_perp_A_d = self.low_prior_down(G_perp_A)

        ll_tokens = self.low_freq_pos(ll).flatten(2).transpose(1, 2).contiguous()
        low_prior_keys = self.low_freq_pos(G_perp_A_d).flatten(2).transpose(1, 2).contiguous()
        low_prior_values = G_perp_A_d.flatten(2).transpose(1, 2).contiguous()
        ll_attended_tokens, _ = self.low_freq_attn(
            ll_tokens,
            low_prior_keys,
            low_prior_values,
            need_weights=False,
        )
        ll_attended = ll_attended_tokens.transpose(1, 2).contiguous().view(B, C, Hd, Wd)

        hf_tokens = torch.cat([hl, lh, hh], dim=1).flatten(2).contiguous()
        high_prior_tokens = A_parallel_G_d.flatten(2).contiguous()
        hf_attended_tokens, _ = self.high_freq_attn(
            hf_tokens,
            high_prior_tokens,
            high_prior_tokens,
            need_weights=False,
        )
        hf_attended = hf_attended_tokens.contiguous().view(B, 3 * C, Hd, Wd)
        hl_attended, lh_attended, hh_attended = torch.chunk(hf_attended, 3, dim=1)

        x_reconstructed = self._idwt(ll_attended, hl_attended, lh_attended, hh_attended, pad_h, pad_w)
        return x + self.out_proj(x_reconstructed)


class AttributeFusionPipeline(nn.Module):
    def __init__(self, img_dim, text_dim=768, stage_key=None, feature_hw=None, attribute_keys=None):
        super().__init__()
        self.dim = img_dim
        self.stage_key = stage_key

        self.feature_extractor = GAEE(
            img_dim=img_dim,
            text_dim=text_dim,
            stage_key=stage_key,
            feature_hw=feature_hw,
            attribute_keys=attribute_keys,
        )
        self.feature_fusion = GAWF(
            img_dim=img_dim,
            text_dim=text_dim,
            feature_hw=feature_hw,
        )

    def forward(self, x, current_global_text, bank_feats_dict):
        attribute_prior = self.feature_extractor(x, bank_feats_dict)
        return self.feature_fusion(x, attribute_prior, current_global_text)

class VisualSemanticCompletion(nn.Module):
    def __init__(self, text_dim, feature_dim=None, num_heads=8, orthogonal_margin=0.1, orthogonal_temperature=0.1, same_attr_pull_weight=0.5):
        super().__init__()
        self.text_dim = text_dim
        self.feature_dim = text_dim if feature_dim is None else feature_dim
        self.num_heads = _resolve_num_heads(self.text_dim, num_heads)
        self.orthogonal_margin = orthogonal_margin
        self.orthogonal_temperature = orthogonal_temperature
        self.same_attr_pull_weight = same_attr_pull_weight
        self.last_orthogonal_loss = None
        self.last_text_contrastive_loss = None
        self.image_pos = ConditionalPositionEncoding2D(self.feature_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.text_dim,
            num_heads=self.num_heads,
            kdim=self.feature_dim,
            vdim=self.feature_dim,
            batch_first=True,
        )
        
        self.out_proj = nn.Linear(self.text_dim, self.text_dim)
        self.norm = nn.LayerNorm(self.text_dim)

    def _pool_attr_text(self, text_tensor):
        return text_tensor.mean(dim=1)

    def _compute_text_contrastive_loss(self, attr_features):
        if len(attr_features) < 1:
            return None

        pooled_features = []
        for feat in attr_features:
            pooled_features.append(self._pool_attr_text(feat))

        attr_matrix = torch.stack(pooled_features, dim=1)
        attr_matrix = F.normalize(attr_matrix, p=2, dim=-1)

        diff_attr_push_loss = attr_matrix.new_tensor(0.0)
        num_attrs = attr_matrix.shape[1]
        if num_attrs > 1:
            attr_sim_matrix = torch.bmm(attr_matrix, attr_matrix.transpose(1, 2))
            off_diag_mask = ~torch.eye(
                num_attrs,
                dtype=torch.bool,
                device=attr_sim_matrix.device,
            ).unsqueeze(0)

            off_diag_sim = attr_sim_matrix.masked_select(off_diag_mask)
            penalty_logits = (off_diag_sim.abs() - self.orthogonal_margin) / self.orthogonal_temperature
            diff_attr_push_loss = F.softplus(penalty_logits).mean() * self.orthogonal_temperature

        same_attr_pull_loss = attr_matrix.new_tensor(0.0)
        batch_size = attr_matrix.shape[0]
        if batch_size > 1:
            same_attr_sim = torch.bmm(
                attr_matrix.transpose(0, 1),
                attr_matrix.transpose(0, 1).transpose(1, 2),
            )
            sample_off_diag_mask = ~torch.eye(
                batch_size,
                dtype=torch.bool,
                device=attr_matrix.device,
            ).unsqueeze(0)
            same_attr_off_diag_sim = same_attr_sim.masked_select(sample_off_diag_mask)
            same_attr_pull_loss = (1.0 - same_attr_off_diag_sim).mean()

        return diff_attr_push_loss + self.same_attr_pull_weight * same_attr_pull_loss

    def forward(self,text_dict, feature_maps):
        self.last_orthogonal_loss = None
        self.last_text_contrastive_loss = None

        G = text_dict['global']
            
        B, N, C = G.shape
        updated_text_dict = {}
        compensated_attr_features = []

        image_keys = self.image_pos(feature_maps).flatten(2).transpose(1, 2).contiguous()
        image_values = feature_maps.flatten(2).transpose(1, 2).contiguous()

        for attr_key, attr_text in text_dict.items():
            if attr_key == 'global':
                updated_text_dict[attr_key] = attr_text
                continue
                
            A = attr_text

            dot_GA = (G * A).sum(dim=-1, keepdim=True)         
            dot_AA = (A * A).sum(dim=-1, keepdim=True) + 1e-6  
            G_parallel_A = (dot_GA / dot_AA) * A               
            G_perp_A = G - G_parallel_A                        

            A_ctx, _ = self.cross_attn(A, image_keys, image_values, need_weights=False)
            

            orthogonal_context_sim = F.cosine_similarity(A_ctx, G_perp_A, dim=-1)

            comp_weight = F.relu(orthogonal_context_sim).unsqueeze(-1)
            
            missing_context = comp_weight * G_perp_A
            
            updated_attr_text = self.norm(A + self.out_proj(missing_context))
                
            updated_text_dict[attr_key] = updated_attr_text
            compensated_attr_features.append(updated_attr_text)

        text_contrastive_loss = self._compute_text_contrastive_loss(compensated_attr_features)
        if text_contrastive_loss is None:
            raise ValueError("No attribute text features were compensated.")
        self.last_text_contrastive_loss = text_contrastive_loss
        self.last_orthogonal_loss = text_contrastive_loss
        
        return updated_text_dict


class MVSC(nn.Module):
    def __init__(self, text_dim, stage_feature_dims, enabled_stage_keys=None, **kwargs):
        super().__init__()
        self.enabled_stage_keys = tuple(enabled_stage_keys)
        self.compensators = nn.ModuleDict({
            stage_key: VisualSemanticCompletion(
                text_dim=text_dim,
                feature_dim=stage_feature_dims[stage_key],
                **kwargs,
            )
            for stage_key in self.enabled_stage_keys
        })
        self.last_orthogonal_loss = None
        self.last_text_contrastive_loss = None
        self._stage_loss_buffer = []

    def begin_forward(self, reference_tensor=None):
        self._stage_loss_buffer = []
        zero = reference_tensor.new_tensor(0.0)
        self.last_orthogonal_loss = zero
        self.last_text_contrastive_loss = zero

    def forward_stage(self, stage_key, text_dict, feature_maps):
        updated_text_dict = self.compensators[stage_key](text_dict, feature_maps)
        stage_loss = self.compensators[stage_key].last_text_contrastive_loss
        self._stage_loss_buffer.append(stage_loss)
        aggregated_loss = torch.stack(self._stage_loss_buffer).mean()
        self.last_text_contrastive_loss = aggregated_loss
        self.last_orthogonal_loss = aggregated_loss
        return updated_text_dict
