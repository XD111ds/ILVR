from trl import SFTTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import deepspeed
from typing import List

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
except Exception:
    apply_multimodal_rotary_pos_emb = None

try:
    import deepspeed
    _HAS_DS = True
except ImportError:
    _HAS_DS = False


class _EMATeacher:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module):
        d = self.decay
        for p_t, p_s in zip(self.teacher.parameters(), student.parameters()):
            p_t.data.mul_(d).add_(p_s.data, alpha=(1.0 - d))


class CustomTrainerStage1(SFTTrainer):
    def __init__(
        self,
        *args,
        sim_weight: float = 1.0,
        ema_tau: float = 0.999,
        coverage_p: float = 0.9,
        image_pool_k: int = 8,
        ce_weight: float = 1.0,
        helper_group_L: int = 256,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sim_weight = float(sim_weight)
        self.coverage_p = float(coverage_p)
        self.helper_group_L = int(helper_group_L)
        self.ce_weight = float(ce_weight)
        self.image_pool_k = int(image_pool_k)
        self._ema = _EMATeacher(self.model, decay=float(ema_tau))

    def _find_latent_segments(self, input_ids, latent_start_id, latent_end_id, latent_pad_id):
        ids = input_ids[0].tolist()
        segments = []
        t = 0
        T = len(ids)
        while t < T:
            if ids[t] == latent_start_id:
                s = t
                e = s + 1
                while e < T and ids[e] != latent_end_id:
                    e += 1
                pad_pos = [i for i in range(s, e+1) if ids[i] == latent_pad_id]
                segments.append(pad_pos)
                t = e + 1
            else:
                t += 1
        return segments

    def _build_firstK_mask(self, input_ids, segments, K_list):
        B, T = input_ids.shape
        mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
        assert len(segments) == len(K_list)
        for pads, K in zip(segments, K_list):
            if K > 0 and len(pads) > 0:
                take = min(K, len(pads))
                for i in pads[:take]:
                    mask[0, i] = True
        return mask

    def _top_p_top_k(self, scores: torch.Tensor, p: float, k: int):
        if scores.numel() == 0 or k <= 0:
            return []
        probs = torch.softmax(scores, dim=0)
        sorted_probs, idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=0)
        pos = torch.nonzero(cum >= p, as_tuple=False)
        if pos.numel() > 0:
            Kp = int(pos[0].item()) + 1
        else:
            Kp = int(probs.numel())
        Kstar = min(Kp, int(k))
        return idx[:Kstar].tolist()

    def _grouped_mean(self, ei: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return ei.new_zeros((0, ei.shape[-1]))
        P = ei.shape[0]
        if P >= k and (P % k) != 0:
            newP = P - (P % k)
            if newP > 0:
                ei = ei[:newP]
            else:
                return ei.new_zeros((k, ei.shape[-1]))
        if ei.shape[0] >= k and ei.shape[0] % k == 0:
            g = ei.view(k, ei.shape[0]//k, ei.shape[-1]).mean(dim=1)
            return g
        if ei.shape[0] == 0:
            return ei.new_zeros((k, ei.shape[-1]))
        rep = math.ceil(k / ei.shape[0])
        g = ei.repeat(rep, 1)[:k]
        return g

    def _maybe_group_for_helper(self, ei: torch.Tensor, L_group: int) -> torch.Tensor:
        if L_group is None or L_group <= 0:
            return ei
        P = int(ei.shape[0])
        if P < L_group:
            return ei
        return self._grouped_mean(ei, L_group)

    def _prefix_text_mean_from_embeds(self, token_embeds: torch.Tensor, input_ids: torch.Tensor,
                                      upto_idx: int, special_ids: set) -> torch.Tensor:
        H = token_embeds.shape[-1]
        t_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if upto_idx > 0:
            t_mask[:, :upto_idx] = 1
        for sid in special_ids:
            t_mask &= (input_ids != sid)
        feats = token_embeds[0, t_mask[0]]
        if feats.numel() == 0:
            feats = token_embeds[0, :max(1, upto_idx)]
        return feats.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def _image_input_global_mean(self, tea, inputs) -> torch.Tensor:
        pv  = inputs.get("pixel_values", None)
        thw = inputs.get("image_grid_thw", None)
        if pv is None or thw is None:
            return None
        pv  = pv.to(next(tea.parameters()).device).type(tea.visual.dtype)
        thw = thw.to(next(tea.parameters()).device)
        patches = tea.visual(pv, grid_thw=thw)
        num_imgs = thw.shape[0]
        if num_imgs == 0 or patches.numel() == 0:
            return None

        s_merge = int(getattr(tea.visual, "spatial_merge_size", 2))
        thw_long = thw.to(dtype=torch.long)
        tokens_per_img = thw_long[:,0] * (thw_long[:,1]//s_merge) * (thw_long[:,2]//s_merge)
        ends = torch.cumsum(tokens_per_img, dim=0).tolist()
        starts = [0] + ends[:-1]
        outs = []
        for st, ed in zip(starts, ends):
            ei = patches[st:ed, :]
            gk = self._grouped_mean(ei, self.image_pool_k)
            outs.append(gk)
        all_groups = torch.cat(outs, dim=0)
        return all_groups.mean(dim=0, keepdim=True)

    def _get_special_ids(self):
        tok = self.tokenizer
        get_id = lambda s: tok(s, return_tensors="pt")["input_ids"][0,0].item()
        latent_pad_id   = get_id("<|latent_pad|>")
        latent_start_id = get_id("<|latent_start|>")
        latent_end_id   = get_id("<|latent_end|>")
        special_ids = {latent_pad_id, latent_start_id, latent_end_id}
        try:
            vision_start_id = get_id("<|vision_start|>")
            vision_end_id   = get_id("<|vision_end|>")
            special_ids.update({vision_start_id, vision_end_id})
        except Exception:
            pass
        img_token_id = getattr(self.model.config, "image_token_id", None)
        if img_token_id is None:
            img_token_id = 151655
        return special_ids, int(img_token_id), int(latent_start_id), int(latent_end_id), int(latent_pad_id)

    @torch.no_grad()
    def _user_side_attn_pooling(self, tea, inputs, ids, attn) -> torch.Tensor:
        device = ids.device
        B, T = ids.shape
        cand_texts = ["<|im_start|>assistant", "<|im_start|>assistant\n"]
        cand_patterns = [
            self.tokenizer(s, return_tensors="pt")["input_ids"][0].to(device) for s in cand_texts
        ]
        start_assistant = -1
        for pat in cand_patterns:
            for i in range(0, T - pat.size(0) + 1):
                if torch.all(ids[0, i:i+pat.size(0)] == pat):
                    start_assistant = i
                    break
            if start_assistant != -1:
                break
        if start_assistant <= 0:
            return None

        special_ids, image_token_id, latent_start_id, latent_end_id, latent_pad_id = self._get_special_ids()
        id_row = ids[0, :start_assistant]
        prompt_idx = [i for i in range(id_row.size(0))
                    if (int(id_row[i].item()) not in special_ids) and (int(id_row[i].item()) != image_token_id)]
        image_idx  = (id_row == image_token_id).nonzero().view(-1).tolist()
        if len(prompt_idx) == 0 or len(image_idx) == 0:
            return None

        old_flag = bool(getattr(tea.config, "output_hidden_states", False))
        tea.config.output_hidden_states = True
        try:
            out = tea(
                input_ids=ids,
                attention_mask=attn,
                pixel_values=inputs.get("pixel_values", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                output_hidden_states=True,
                return_dict=True,
            )
            hs = out.hidden_states
        finally:
            tea.config.output_hidden_states = old_flag

        if isinstance(hs, (tuple, list)):
            H_last = hs[-1]
            H_pre  = hs[-2] if len(hs) > 1 else hs[-1]
        else:
            H_last = hs
            H_pre  = hs

        try:
            layer_last = tea.model.layers[-1].self_attn
            q = layer_last.q_proj(H_pre)
            k = layer_last.k_proj(H_pre)
            num_heads = layer_last.num_heads
            dh = q.shape[-1] // num_heads
            q = q.view(1, T, num_heads, dh).transpose(1, 2)
            kv = layer_last.num_key_value_heads
            k = k.view(1, T, kv, dh).transpose(1, 2)
            if kv != num_heads:
                rep = num_heads // kv
                k = k[:, :, None, :, :].expand(1, kv, rep, T, dh).reshape(1, num_heads, T, dh)

            q_sel = q[:, :, prompt_idx, :]
            k_sel = k[:, :, image_idx, :]
            logits = torch.einsum("bhpd,bhqd->bhpq", q_sel, k_sel) / (dh ** 0.5)
            probs_per_prompt = torch.softmax(logits, dim=-1)
            w = probs_per_prompt.mean(dim=2).mean(dim=1).squeeze(0)
            w = w / (w.sum() + 1e-9)

            V_img = H_last[0, :start_assistant, :][image_idx, :]
            V_prm = H_last[0, :start_assistant, :][prompt_idx, :]
            r_img = (w.unsqueeze(0) @ V_img).squeeze(0)
            r_txt = V_prm.mean(dim=0)
            u = torch.stack([r_img, r_txt], dim=0).mean(dim=0, keepdim=True)
            return u
        except Exception:
            H_sub = H_last[0, :start_assistant, :]
            V_img = H_sub[image_idx, :]
            V_prm = H_sub[prompt_idx, :]
            sim = (V_prm @ V_img.t()) / (V_img.shape[-1] ** 0.5)
            w = torch.softmax(sim, dim=-1).mean(dim=0)
            w = w / (w.sum() + 1e-9)
            r_img = (w.unsqueeze(0) @ V_img).squeeze(0)
            r_txt = V_prm.mean(dim=0)
            u = torch.stack([r_img, r_txt], dim=0).mean(dim=0, keepdim=True)
            return u

    def _extract_assistant_text_spans(self, ids: torch.Tensor,
                                      latent_starts: List[int], latent_ends: List[int],
                                      assistant_start: int, special_ids: set) -> List[List[int]]:
        spans = []
        prev = assistant_start
        for s, e in zip(latent_starts, latent_ends):
            span_idx = []
            for t in range(prev, s):
                tid = int(ids[t].item())
                if (tid not in special_ids):
                    span_idx.append(t)
            spans.append(span_idx)
            prev = e + 1
        return spans

    @torch.no_grad()
    def _teacher_build_latents(self, inputs, k, p):
        device = self.model.device
        ids  = inputs["input_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        tea  = self._ema.teacher.to(device).eval()

        special_ids, image_token_id, latent_start_id, latent_end_id, latent_pad_id = self._get_special_ids()

        seg_pad_indices = self._find_latent_segments(ids, latent_start_id, latent_end_id, latent_pad_id)
        if len(seg_pad_indices) == 0:
            return None, torch.zeros_like(ids, dtype=torch.bool)

        idlist = ids[0].tolist()
        starts = []
        ends   = []
        t = 0; T = len(idlist)
        while t < T:
            if idlist[t] == latent_start_id:
                s = t
                e = s + 1
                while e < T and idlist[e] != latent_end_id:
                    e += 1
                starts.append(s); ends.append(e)
                t = e + 1
            else:
                t += 1

        pat = self.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0].to(device)
        start_assistant = -1
        for i in range(0, ids.size(1) - pat.size(0) + 1):
            if torch.all(ids[0, i:i+pat.size(0)] == pat):
                start_assistant = i; break
        if start_assistant <= 0:
            return None, torch.zeros_like(ids, dtype=torch.bool)

        u = self._user_side_attn_pooling(tea, inputs, ids, attn)
        if u is None:
            token_embeds = tea.get_input_embeddings()(ids)
            u_text = self._prefix_text_mean_from_embeds(token_embeds, ids, start_assistant, special_ids)
            u_img  = self._image_input_global_mean(tea, inputs)
            parts = [u_text] + ([u_img] if u_img is not None else [])
            u = torch.stack([x.squeeze(0) for x in parts]).mean(dim=0, keepdim=True)

        pv  = inputs.get("pixel_values_latent", None)
        thw = inputs.get("image_grid_thw_latent", None)
        if pv is None or thw is None:
            return None, torch.zeros_like(ids, dtype=torch.bool)
        pv  = pv.to(device).to(tea.visual.dtype)
        thw = thw.to(device)
        patch_all = tea.visual(pv, grid_thw=thw)
        num_imgs  = int(thw.shape[0])

        s_merge = int(getattr(tea.visual, "spatial_merge_size", 2))
        thw_long = thw.to(dtype=torch.long)
        tokens_per_img = (thw_long[:,0] * (thw_long[:,1]//s_merge) * (thw_long[:,2]//s_merge))
        ends_img = torch.cumsum(tokens_per_img, dim=0).tolist()
        starts_img = [0] + ends_img[:-1]
        slices_per_img = [(int(st), int(ed)) for st, ed in zip(starts_img, ends_img)]
        assert len(slices_per_img) == num_imgs

        text_spans = self._extract_assistant_text_spans(ids[0], starts, ends, start_assistant, special_ids)
        L_group = getattr(self, "helper_group_L", None)
        if L_group is None:
            L_group = int(self.image_pool_k)

        latents_list = []
        Kstars = []
        prev_sel_mean = None

        for seg_idx, pad_pos in enumerate(seg_pad_indices):
            if len(pad_pos) == 0:
                Kstars.append(0); continue

            text_idx = text_spans[seg_idx] if seg_idx < len(text_spans) else []
            old_flag = bool(getattr(tea.config, "output_hidden_states", False))
            tea.config.output_hidden_states = True
            try:
                out_assist = tea(
                    input_ids=ids,
                    attention_mask=attn,
                    pixel_values=inputs.get("pixel_values", None),
                    image_grid_thw=inputs.get("image_grid_thw", None),
                    output_hidden_states=True,
                    return_dict=True,
                )
            finally:
                tea.config.output_hidden_states = old_flag

            hs = out_assist.hidden_states
            x = hs[-1] if isinstance(hs, (tuple, list)) else hs
            H_last_2d = x[0] if x.dim() == 3 else (x if x.dim() == 2 else tea.get_input_embeddings()(ids)[0])

            q_parts = [u]

            if len(text_idx) > 0:
                idx_tensor = torch.tensor(text_idx, device=H_last_2d.device, dtype=torch.long)
                q_parts.append(H_last_2d.index_select(0, idx_tensor).mean(dim=0, keepdim=True))

            if prev_sel_mean is not None:
                q_parts.append(prev_sel_mean)

            q_t = torch.stack([x.squeeze(0) for x in q_parts], dim=0).mean(dim=0, keepdim=True)

            assert seg_idx < num_imgs, "latent 段数量与 helper images 数量不一致"
            st_img, ed_img = slices_per_img[seg_idx]
            ei = patch_all[st_img:ed_img, :]
            cand = self._maybe_group_for_helper(ei, L_group)

            sim = F.cosine_similarity(cand, q_t.expand_as(cand), dim=-1)
            top_idx = self._top_p_top_k(sim, p=self.coverage_p, k=k)
            Kstar = len(top_idx)
            Kstars.append(Kstar)

            if Kstar > 0:
                chosen = cand[top_idx[:Kstar], :]
                latents_list.append(chosen)
                prev_sel_mean = chosen.mean(dim=0, keepdim=True)

        if len(latents_list) == 0:
            return None, torch.zeros_like(ids, dtype=torch.bool)
        latents = torch.cat(latents_list, dim=0).unsqueeze(0)

        firstK_mask = self._build_firstK_mask(ids, seg_pad_indices, Kstars)
        return latents, firstK_mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        k = getattr(self.model.config, "latent_size", 8)
        teacher_latents, firstK_mask = self._teacher_build_latents(inputs, k=k, p=self.coverage_p)
        
        with torch.no_grad():
            pv_lat = inputs.get("pixel_values_latent", None)
            thw_lat = inputs.get("image_grid_thw_latent", None)
            if pv_lat is not None and thw_lat is not None:
                _ = self.model.visual(
                    pv_lat.to(self.model.device).to(self.model.visual.dtype),
                    grid_thw=thw_lat.to(self.model.device)
                )
        
        if teacher_latents is None:
            ce_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            return (ce_loss, outputs) if return_outputs else ce_loss

        if teacher_latents.dim() == 2:
            teacher_latents = teacher_latents.unsqueeze(0)
        S = int(firstK_mask.sum().item())
        if teacher_latents.shape[1] != S:
            ce_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            return (ce_loss, outputs) if return_outputs else ce_loss

        mod_inputs = dict(inputs)
        mod_inputs.pop("pixel_values_latent", None)
        mod_inputs["latent_hidden_states"] = teacher_latents.to(self.model.device).to(self.model.dtype)
        mod_inputs["image_out_mask"] = firstK_mask

        ce_loss, outputs = super().compute_loss(
            model, mod_inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if self.sim_weight == 0.0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        pred_h = outputs.hidden_states
        inp_h  = outputs.inputs_embeds
        B, T, H = pred_h.shape
        if T <= 1:
            return (ce_loss, outputs) if return_outputs else ce_loss

        mask = mod_inputs["image_out_mask"][:, -(T - 1):].to(pred_h.device).bool()
        if not mask.any():
            return (ce_loss, outputs) if return_outputs else ce_loss

        pred = pred_h[..., :-1, :][mask].contiguous().float()
        gt   = inp_h[...,  1:, :][mask].contiguous().detach().float()
        gt = gt + 0.01 * torch.randn_like(gt)

        cos = F.cosine_similarity(gt, pred, dim=-1).mean()
        sim_loss = 1.0 - cos
        loss = self.ce_weight * ce_loss + self.sim_weight * sim_loss
        return (loss, outputs) if return_outputs else loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if not hasattr(self, "_ema") or self._ema is None:
            return
        d = float(self._ema.decay)
        with torch.no_grad():
            if _HAS_DS and any(hasattr(p, "ds_id") for p in self.model.parameters()):
                for p_t, p_s in zip(self._ema.teacher.parameters(), self.model.parameters()):
                    with deepspeed.zero.GatheredParameters(p_s, modifier_rank=0):
                        if p_s.data.numel() == 0:
                            continue
                        p_data = p_s.data
                        if p_data.device != p_t.data.device:
                            p_data = p_data.to(p_t.data.device)
                        p_t.data.mul_(d).add_(p_data, alpha=(1.0 - d))
            else:
                self._ema.update(self.model)
