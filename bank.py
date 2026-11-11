import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

FUSION_STAGE_KEYS = ("x0", "x1", "x2", "x3")

class ADPM:
    def __init__(self, attribute_keys, eps=0.01, pending_eps=0.05, min_samples=2, 
                 momentum=0.99, sim_threshold=0.90, max_clusters=30, device='cuda',
                 init_temperature=0.1, ema_max_alpha=0.1,
                 pending_recluster_size=32, pending_keep_noise=16):
        self.attribute_keys = attribute_keys
        self.eps = eps
        self.pending_eps = pending_eps
        self.min_samples = min_samples
        self.momentum = momentum
        self.sim_threshold = sim_threshold
        self.max_clusters = max_clusters
        self.device = device
        self.init_temperature = init_temperature
        self.ema_max_alpha = ema_max_alpha
        self.pending_recluster_size = pending_recluster_size
        self.pending_keep_noise = pending_keep_noise
        
        self.bank = {}    
        self.text_centers = {}  
        self.rep_text_feats = {}
        self.rep_global_feats = {}
        self.hit_counts = {}
        self.pending_samples = defaultdict(dict)


    def _normalize_stage_features(self, features):
        stage_features = {
            stage_key: features[stage_key]
            for stage_key in FUSION_STAGE_KEYS
            if stage_key in features
        }

        missing_stages = [stage_key for stage_key in FUSION_STAGE_KEYS if stage_key not in stage_features]
        if missing_stages:
            raise ValueError(f"Missing stage features for: {missing_stages}")
        return stage_features

    def _pool_text(self, text_tensor):
        if text_tensor.dim() != 2:
            raise ValueError(f"Expected token text with shape [N, C], got {tuple(text_tensor.shape)}.")
        return text_tensor.mean(dim=0)

    def _dynamic_ema_alpha(self, attr_key, cluster_idx):
        hit_count = self.hit_counts[attr_key][cluster_idx].float()
        min_alpha = max(1.0 - self.momentum, 1e-6)
        max_alpha = max(self.ema_max_alpha, min_alpha)
        alpha = 1.0 / (hit_count + 1.0)
        alpha = torch.clamp(
            alpha,
            min=min_alpha,
            max=max_alpha,
        )
        return alpha

    def _normalize_sample_ids(self, sample_ids, batch_size):
        sample_ids = sample_ids.detach().cpu().tolist()

        sample_keys = [str(sample_id) for sample_id in sample_ids]
        if len(sample_keys) != batch_size:
            raise ValueError(f"sample_ids length {len(sample_keys)} does not match batch size {batch_size}.")
        return sample_keys

    def _add_pending_sample(self, attr_key, sample_key, cur_text_center, cur_text_tokens, cur_img_feat, cur_global_feat):
        pending = self.pending_samples[attr_key]
        if sample_key in pending:
            pending.pop(sample_key)

        pending[sample_key] = {
            "text_center": cur_text_center.detach().float().cpu(),
            "text_tokens": cur_text_tokens.detach().float().cpu(),
            "image": {k: v.detach().float().cpu() for k, v in cur_img_feat.items()},
            "global": cur_global_feat.detach().float().cpu(),
        }

    def _queue_pending_sample(self, attr_key, sample_key, cur_text_center, cur_text_tokens, cur_img_feat, cur_global_feat):
        self._add_pending_sample(
            attr_key,
            sample_key,
            cur_text_center,
            cur_text_tokens,
            cur_img_feat,
            cur_global_feat,
        )
        if len(self.pending_samples[attr_key]) >= self.pending_recluster_size:
            self.recluster_pending(attr_key)

    def _weighted_cluster_prototype(self, text_centers, text_tokens_list, img_feats_list, global_feats_list):
        text_matrix = torch.stack([feat.float().cpu() for feat in text_centers])
        center = text_matrix.mean(dim=0)
        center = F.layer_norm(center, (center.shape[-1],))

        denom = text_matrix.norm(dim=1) * (center.norm() + 1e-8) + 1e-8
        similarities = torch.mv(text_matrix, center) / denom
        logits = similarities / max(self.init_temperature, 1e-8)
        weights = torch.softmax(logits, dim=0)

        weighted_center = (text_matrix * weights.view(-1, 1)).sum(dim=0)
        weighted_center = F.layer_norm(weighted_center, (weighted_center.shape[-1],))

        token_matrix = torch.stack([feat.float().cpu() for feat in text_tokens_list])
        weighted_tokens = (token_matrix * weights.view(-1, 1, 1)).sum(dim=0)
        weighted_tokens = F.layer_norm(weighted_tokens, (weighted_tokens.shape[-1],))

        weighted_images = {}
        for stage_key in FUSION_STAGE_KEYS:
            stage_maps = [img_feats[stage_key].float().cpu() for img_feats in img_feats_list if stage_key in img_feats]
            if len(stage_maps) != len(img_feats_list):
                raise ValueError(f"Missing stage '{stage_key}' while building weighted cluster prototype.")
            stacked_stage_maps = torch.stack(stage_maps)
            view_shape = (weights.shape[0],) + (1,) * (stacked_stage_maps.dim() - 1)
            weighted_images[stage_key] = (
                stacked_stage_maps * weights.view(view_shape)
            ).sum(dim=0)

        stacked_globals = torch.stack([global_feat.float().cpu() for global_feat in global_feats_list])
        view_shape = (weights.shape[0],) + (1,) * (stacked_globals.dim() - 1)
        weighted_global = (stacked_globals * weights.view(view_shape)).sum(dim=0)

        return weighted_center, weighted_tokens, weighted_images, weighted_global

    def _ema_update_cluster(self, attr_key, cluster_idx, text_center, text_tokens, img_proto, global_proto, alpha=0.01, hit_count=1.0):
        device = self.text_centers[attr_key].device
        missing_stage_keys = [stage_key for stage_key in FUSION_STAGE_KEYS if stage_key not in img_proto]
        if missing_stage_keys:
            raise ValueError(f"Missing image prototypes for stages: {missing_stage_keys}")

        text_center = text_center.to(device)
        text_tokens = text_tokens.to(device)
        alpha = torch.as_tensor(alpha, dtype=torch.float32, device=device)
        keep = 1.0 - alpha

        new_center = keep * self.text_centers[attr_key][cluster_idx] + alpha * text_center
        new_center = F.layer_norm(new_center, (new_center.shape[-1],))
        self.text_centers[attr_key][cluster_idx] = new_center
        new_tokens = keep * self.rep_text_feats[attr_key][cluster_idx] + alpha * text_tokens
        self.rep_text_feats[attr_key][cluster_idx] = F.layer_norm(new_tokens, (new_tokens.shape[-1],))

        for stage_key, stage_feat in img_proto.items():
            stage_feat = stage_feat.to(device)
            old_stage_feat = self.bank[attr_key][stage_key][cluster_idx]
            self.bank[attr_key][stage_key][cluster_idx] = keep * old_stage_feat + alpha * stage_feat

        global_proto = global_proto.to(device)
        old_global_feat = self.rep_global_feats[attr_key][cluster_idx]
        self.rep_global_feats[attr_key][cluster_idx] = keep * old_global_feat + alpha * global_proto

        self.hit_counts[attr_key][cluster_idx] += float(hit_count)

    def _append_cluster(self, attr_key, text_center, text_tokens, img_proto, global_proto, hit_count=1.0):
        missing_stage_keys = [stage_key for stage_key in FUSION_STAGE_KEYS if stage_key not in img_proto]
        if missing_stage_keys:
            raise ValueError(f"Missing image prototypes for stages: {missing_stage_keys}")
        text_center = text_center.to(self.device)
        text_tokens = text_tokens.to(self.device)
        img_proto = {stage_key: stage_feat.to(self.device) for stage_key, stage_feat in img_proto.items()}
        global_proto = global_proto.to(self.device)
        hit_tensor = torch.tensor([float(hit_count)], dtype=torch.float32, device=self.device)

        if attr_key not in self.text_centers:
            self.text_centers[attr_key] = text_center.unsqueeze(0)
            self.rep_text_feats[attr_key] = text_tokens.unsqueeze(0)
            self.bank[attr_key] = {stage_key: stage_feat.unsqueeze(0) for stage_key, stage_feat in img_proto.items()}
            self.rep_global_feats[attr_key] = global_proto.unsqueeze(0)
            self.hit_counts[attr_key] = hit_tensor
            return

        self.text_centers[attr_key] = torch.cat([self.text_centers[attr_key], text_center.unsqueeze(0)], dim=0)
        self.rep_text_feats[attr_key] = torch.cat([self.rep_text_feats[attr_key], text_tokens.unsqueeze(0)], dim=0)
        for stage_key, stage_feat in img_proto.items():
            self.bank[attr_key][stage_key] = torch.cat([self.bank[attr_key][stage_key], stage_feat.unsqueeze(0)], dim=0)

        self.rep_global_feats[attr_key] = torch.cat([self.rep_global_feats[attr_key], global_proto.unsqueeze(0)], dim=0)
        self.hit_counts[attr_key] = torch.cat([self.hit_counts[attr_key], hit_tensor], dim=0)

    def _set_cluster(self, attr_key, cluster_idx, text_center, text_tokens, img_proto, global_proto):
        missing_stage_keys = [stage_key for stage_key in FUSION_STAGE_KEYS if stage_key not in img_proto]
        if missing_stage_keys:
            raise ValueError(f"Missing image prototypes for stages: {missing_stage_keys}")

        self.text_centers[attr_key][cluster_idx] = text_center.to(self.device)
        self.rep_text_feats[attr_key][cluster_idx] = text_tokens.to(self.device)

        for stage_key, stage_feat in img_proto.items():
            stage_feat = stage_feat.to(self.device)
            self.bank[attr_key][stage_key][cluster_idx] = stage_feat

        self.rep_global_feats[attr_key][cluster_idx] = global_proto.to(self.device)

    def _merge_or_add_cluster(self, attr_key, text_center, text_tokens, img_proto, global_proto, hit_count=1.0):
        if attr_key not in self.text_centers:
            self._append_cluster(attr_key, text_center, text_tokens, img_proto, global_proto, hit_count)
            return "created"

        text_center = text_center.to(self.text_centers[attr_key].device)
        similarities = F.cosine_similarity(text_center.unsqueeze(0), self.text_centers[attr_key])
        max_sim, best_idx = torch.max(similarities, dim=0)
        match_idx = best_idx.item()

        if max_sim.item() >= self.sim_threshold:
            old_hit = self.hit_counts[attr_key][match_idx].float()
            min_alpha = max(1.0 - self.momentum, 1e-6)
            max_alpha = max(self.ema_max_alpha, min_alpha)
            alpha = torch.tensor(float(hit_count), dtype=torch.float32, device=self.device)
            alpha = alpha / (old_hit + alpha)
            alpha = torch.clamp(alpha, min=min_alpha, max=max_alpha)
            self._ema_update_cluster(
                attr_key,
                match_idx,
                text_center,
                text_tokens,
                img_proto,
                global_proto,
                alpha=alpha,
                hit_count=hit_count,
            )
            return "merged"

        if self.text_centers[attr_key].shape[0] < self.max_clusters:
            self._append_cluster(attr_key, text_center, text_tokens, img_proto, global_proto, hit_count)
            return "added"

        dead_idx = torch.argmin(self.hit_counts[attr_key]).item()
        self._set_cluster(attr_key, dead_idx, text_center, text_tokens, img_proto, global_proto)
        self.hit_counts[attr_key][dead_idx] = float(hit_count)
        return "replaced"

    @torch.no_grad()
    def recluster_pending(self, attr_key):
        target_keys = [attr_key]
        report = {}

        for key in target_keys:
            pending = self.pending_samples.get(key, {})
            pending_count = len(pending)

            pending_items = list(pending.items())
            text_matrix = np.stack([record["text_center"].numpy() for _, record in pending_items])
            labels = DBSCAN(eps=self.pending_eps, min_samples=self.min_samples, metric="cosine").fit_predict(text_matrix)

            formed_clusters = 0
            clustered_indices = set()
            for label in sorted(set(labels)):
                if label == -1:
                    continue

                cluster_indices = [i for i, cur_label in enumerate(labels) if cur_label == label]
                text_centers = [pending_items[i][1]["text_center"] for i in cluster_indices]
                text_tokens_list = [pending_items[i][1]["text_tokens"] for i in cluster_indices]
                img_feats_list = [pending_items[i][1]["image"] for i in cluster_indices]
                global_feats_list = [pending_items[i][1]["global"] for i in cluster_indices]
                text_center, text_tokens, img_proto, global_proto = self._weighted_cluster_prototype(
                    text_centers,
                    text_tokens_list,
                    img_feats_list,
                    global_feats_list,
                )
                self._merge_or_add_cluster(
                    key,
                    text_center,
                    text_tokens,
                    img_proto,
                    global_proto,
                    hit_count=len(cluster_indices),
                )
                clustered_indices.update(cluster_indices)
                formed_clusters += 1

            noise_items = [pending_items[i] for i, cur_label in enumerate(labels) if cur_label == -1 or i not in clustered_indices]
            new_pending = {}
            for sample_key, record in noise_items[-self.pending_keep_noise:]:
                new_pending[sample_key] = record
            self.pending_samples[key] = new_pending
            report[key] = {"pending_before": pending_count, "clusters": formed_clusters, "pending_after": len(new_pending)}

        return report


    def build_from_dataloader(self, dataloader, feature_extractor):
        text_features_dict = defaultdict(list)
        text_token_features_dict = defaultdict(list)
        img_features_dict = defaultdict(list)
        global_features_dict = defaultdict(list)

        print("Collecting prototype bank initialization features...")
        with torch.no_grad():
            for sampled_batch in dataloader:
                raw_images = sampled_batch["image"]
                text_attr_dict = sampled_batch["text"]

                for k, v in text_attr_dict.items():
                    text_attr_dict[k] = F.layer_norm(v, (v.shape[-1],))

                img_features = self._normalize_stage_features(feature_extractor(raw_images))
                img_features_cpu = {stage_key: feat.detach().cpu() for stage_key, feat in img_features.items()}
                first_stage = next(iter(img_features_cpu.values()))
                batch_size = first_stage.shape[0]

                for b in range(batch_size):
                    cur_global = text_attr_dict["global"][b].cpu().numpy()
                    for attr_key in self.attribute_keys:
                        if attr_key == "global":
                            continue
                        text_tokens = text_attr_dict[attr_key][b].detach().float().cpu()
                        text_feat = self._pool_text(text_tokens).numpy()
                        text_features_dict[attr_key].append(text_feat)
                        text_token_features_dict[attr_key].append(text_tokens)
                        img_features_dict[attr_key].append({
                            stage_key: stage_feats[b]
                            for stage_key, stage_feats in img_features_cpu.items()
                        })
                        global_features_dict[attr_key].append(cur_global)

        print("Building prototype clusters...")
        for attr_key in self.attribute_keys:
            if attr_key == "global":
                continue

            text_matrix = np.array(text_features_dict[attr_key])
            labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine").fit_predict(text_matrix)

            clusters_img_maps = defaultdict(list)
            clusters_text_centers = []
            clusters_rep_texts = []
            clusters_rep_globals = []
            hit_counts_list = []

            for label in set(labels):
                if label == -1:
                    continue

                cluster_text_feats = np.array([text_features_dict[attr_key][i] for i, cur_label in enumerate(labels) if cur_label == label])
                cluster_text_tokens = torch.stack([
                    text_token_features_dict[attr_key][i]
                    for i, cur_label in enumerate(labels)
                    if cur_label == label
                ])
                cluster_img_maps_list = [img_features_dict[attr_key][i] for i, cur_label in enumerate(labels) if cur_label == label]
                cluster_global_list = [global_features_dict[attr_key][i] for i, cur_label in enumerate(labels) if cur_label == label]

                center = np.mean(cluster_text_feats, axis=0)
                center_tensor = torch.tensor(center, dtype=torch.float32)
                center = F.layer_norm(center_tensor, (center_tensor.shape[-1],)).numpy()

                denom = np.linalg.norm(cluster_text_feats, axis=1) * (np.linalg.norm(center) + 1e-8) + 1e-8
                similarities = np.dot(cluster_text_feats, center) / denom
                logits = similarities / max(self.init_temperature, 1e-8)
                logits = logits - np.max(logits)
                weights = np.exp(logits)
                weights = weights / (np.sum(weights) + 1e-8)
                weight_tensor = torch.tensor(weights, dtype=torch.float32)

                weighted_text = torch.tensor(np.sum(cluster_text_feats * weights[:, None], axis=0), dtype=torch.float32)
                weighted_text = F.layer_norm(weighted_text, (weighted_text.shape[-1],))
                weighted_text_tokens = (
                    cluster_text_tokens * weight_tensor.view(-1, 1, 1)
                ).sum(dim=0)
                weighted_text_tokens = F.layer_norm(
                    weighted_text_tokens,
                    (weighted_text_tokens.shape[-1],),
                )
                clusters_text_centers.append(weighted_text)
                clusters_rep_texts.append(weighted_text_tokens)

                for stage_key in FUSION_STAGE_KEYS:
                    stage_maps = [img_maps[stage_key].float() for img_maps in cluster_img_maps_list if stage_key in img_maps]
                    if len(stage_maps) != len(cluster_img_maps_list):
                        raise ValueError(f"Missing stage '{stage_key}' while building cluster for attribute '{attr_key}'.")
                    stacked_stage_maps = torch.stack(stage_maps)
                    view_shape = (weight_tensor.shape[0],) + (1,) * (stacked_stage_maps.dim() - 1)
                    weighted_stage_map = (stacked_stage_maps * weight_tensor.view(view_shape)).sum(dim=0)
                    clusters_img_maps[stage_key].append(weighted_stage_map)

                stacked_globals = torch.stack([torch.tensor(global_feat, dtype=torch.float32) for global_feat in cluster_global_list])
                view_shape = (weight_tensor.shape[0],) + (1,) * (stacked_globals.dim() - 1)
                weighted_global = (stacked_globals * weight_tensor.view(view_shape)).sum(dim=0)
                clusters_rep_globals.append(weighted_global)
                hit_counts_list.append(float(len(cluster_text_feats)))

            if clusters_img_maps:
                self.bank[attr_key] = {
                    stage_key: torch.stack(stage_maps).to(self.device)
                    for stage_key, stage_maps in clusters_img_maps.items()
                }
                self.text_centers[attr_key] = torch.stack(clusters_text_centers).to(self.device)
                self.rep_text_feats[attr_key] = torch.stack(clusters_rep_texts).to(self.device)
                if len(clusters_rep_globals) != len(clusters_text_centers):
                    raise ValueError(f"Global cluster count mismatch for attribute '{attr_key}'.")
                self.rep_global_feats[attr_key] = torch.stack(clusters_rep_globals).to(self.device)
                self.hit_counts[attr_key] = torch.tensor(hit_counts_list, dtype=torch.float32).to(self.device)
                cluster_count = next(iter(self.bank[attr_key].values())).shape[0]
                print(f"Attribute [{attr_key}]: generated {cluster_count} clusters.")

    def query_all_features(self, query_features_dict):
        retrieved_results = {}
        for attr_key, query_feat in query_features_dict.items():
            if attr_key == "global":
                continue
                
            centers = self.text_centers[attr_key].detach().clone()
            
            pooled_queries = []
            for b in range(query_feat.shape[0]):
                pooled_queries.append(self._pool_text(query_feat[b]))
            query_matrix = torch.stack(pooled_queries, dim=0).to(centers.device)
            
            similarities = F.cosine_similarity(query_matrix.unsqueeze(1), centers.unsqueeze(0), dim=-1)
            best_match_idx = torch.argmax(similarities, dim=1)
            
            global_feat_retrieved = self.rep_global_feats[attr_key][best_match_idx].detach().clone()
                
            bank_images = self.bank[attr_key]
            retrieved_images = {stage_key: stage_feats[best_match_idx].detach().clone() for stage_key, stage_feats in bank_images.items()}

            retrieved_results[attr_key] = {'image': retrieved_images, 'text': self.rep_text_feats[attr_key][best_match_idx].detach().clone(), 'global': global_feat_retrieved}
            
        return retrieved_results

    @torch.no_grad()
    def update_bank(self, detached_text_dict, fused_img_features, sample_ids=None):
        for stage_key, stage_feats in fused_img_features.items():
            if stage_feats.dim() != 4:
                raise ValueError(f"update_bank expects {stage_key} features with shape [B, C, H, W].")

        detached_img_features = {stage_key: stage_feats.detach().to(self.device) for stage_key, stage_feats in fused_img_features.items()}
        first_stage = next(iter(detached_img_features.values()))
        batch_size = first_stage.shape[0]
        sample_keys = self._normalize_sample_ids(sample_ids, batch_size)
        
        for b in range(batch_size):
            sample_key = sample_keys[b]
            cur_img_feat = {stage_key: stage_feats[b] for stage_key, stage_feats in detached_img_features.items()}
            
            cur_global_feat = detached_text_dict["global"][b]
            
            for attr_key, text_feats in detached_text_dict.items():
                if attr_key == "global":
                    continue
                    
                cur_text_tokens = text_feats[b].detach().float()
                cur_text_center = self._pool_text(cur_text_tokens).squeeze()
                
                cur_text_center = F.layer_norm(cur_text_center, (cur_text_center.shape[-1],))

                if attr_key not in self.text_centers:
                    self._queue_pending_sample(
                        attr_key,
                        sample_key,
                        cur_text_center,
                        cur_text_tokens,
                        cur_img_feat,
                        cur_global_feat,
                    )
                    continue
                
                centers = self.text_centers[attr_key] 
                
                similarities = F.cosine_similarity(cur_text_center.unsqueeze(0), centers)
                max_sim, best_idx = torch.max(similarities, dim=0)
                
                if max_sim.item() < self.sim_threshold:
                    self._queue_pending_sample(
                        attr_key,
                        sample_key,
                        cur_text_center,
                        cur_text_tokens,
                        cur_img_feat,
                        cur_global_feat,
                    )
                else:
                    match_idx = best_idx.item()
                    ema_alpha = self._dynamic_ema_alpha(attr_key, match_idx)
                    self._ema_update_cluster(
                        attr_key,
                        match_idx,
                        cur_text_center,
                        cur_text_tokens,
                        cur_img_feat,
                        cur_global_feat,
                        alpha=ema_alpha,
                        hit_count=1.0,
                    )

    def get_cluster_counts(self):
        cluster_counts = {}
        for attr_key in self.attribute_keys:
            if attr_key == "global":
                continue
            if attr_key not in self.text_centers:
                cluster_counts[attr_key] = 0
                continue
            cluster_counts[attr_key] = int(self.text_centers[attr_key].shape[0])
        return cluster_counts

    def save(self, filepath):
        save_dir = os.path.dirname(filepath)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dict = {
            'bank': self.bank,
            'text_centers': self.text_centers,
            'rep_text_feats': self.rep_text_feats,
            'rep_global_feats': self.rep_global_feats, 
            'hit_counts': self.hit_counts
        }
        torch.save(save_dict, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.bank = checkpoint['bank']
        self.text_centers = checkpoint['text_centers']
        self.rep_text_feats = checkpoint['rep_text_feats']
        self.rep_global_feats = checkpoint['rep_global_feats'] 
        self.hit_counts = checkpoint['hit_counts']
        for attr_key, token_prototypes in self.rep_text_feats.items():
            if token_prototypes.dim() != 3:
                raise ValueError(
                    f"Attribute '{attr_key}' requires token prototypes [K, N, C], "
                    f"got {tuple(token_prototypes.shape)}. Rebuild the NYL62 prototype bank."
                )

    def to(self, device):
        self.device = device
        for key in self.bank:
            self.bank[key] = {
                stage_key: stage_feats.to(device)
                for stage_key, stage_feats in self.bank[key].items()
            }
            self.text_centers[key] = self.text_centers[key].to(device)
            self.rep_text_feats[key] = self.rep_text_feats[key].to(device)
            self.rep_global_feats[key] = self.rep_global_feats[key].to(device)
            self.hit_counts[key] = self.hit_counts[key].to(device)
