import torch
import logging
import numpy as np
import json
import random
import os
import argparse
from datasets import Dataset
from transformers import LogitsProcessor

from transformers import LogitsProcessor
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_output_path", type=str, default="evaluation_results.json", help="Path to save evaluation results JSON file.")
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument("--stage", type=str, default="stage1")
    parser.add_argument("--task", type=str, default="vsp-spatial-reasoning", choices=["vsp-spatial-reasoning", "vsp-spatial-planning", "blink-jigsaw", "sat","vsp-spatial-planning-cot",'zebra-cot'])
    parser.add_argument("--test_data_path", type=str, default=None, help="Path to the test data jsonl file for evaluation after training.")
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=612)
    parser.add_argument("--compress_strategy", type=str, default='average', choices=['average'])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--data_path", type=str, default='PathToJsonlData')
    parser.add_argument("--log_file", type=str, default='./log.txt')
    parser.add_argument("--save_model_path", type=str, default='./checkpoints/model_stage1')
    parser.add_argument("--load_model_path", type=str, default='./checkpoints/model_stage1')
    parser.add_argument("--cache_dir", type=str, default='./cache')
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--warm_up_steps", type=int, default=10)
    parser.add_argument("--sim_weight", type=float, default=1.0)
    parser.add_argument("--ema_tau", type=float, default=0.999)
    parser.add_argument("--coverage_p", type=float, default=0.9)
    parser.add_argument("--image_pool_k", type=int, default=8)  
    parser.add_argument("--latent_ce_ratio", type=float, default=0.0, help="stage2")
    parser.add_argument("--use_lora", action="store_true", help="")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    parser.add_argument("--lora_target_modules", type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")
    
    parser.add_argument(
        "--resume_from_checkpoint_path",
        type=str,
        default=None,
        help="Path to a specific checkpoint to resume training from. If None, training starts from scratch."
    )
    parser.add_argument("--helper_group_L", type=int, default=256,help="")
    parser.add_argument(
        "--latent_num_segments_train", type=int, default=8,
        help="Stage-2"
    )
    parser.add_argument("--max_seq_length_train", type=int, default=8192,
        help="Stage-2")  # 改默认 8192
    # global_pad_to 已存在，保留
# 其余不变

    return parser.parse_args()

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_jsonl_dataset(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        data = data[:]
    return Dataset.from_list(data)

def place_input_image(
    text,
    image_pad="<|vision_start|><|image_pad|><|vision_end|>",
    image_placeholder="<image>",
    sep_token="<|im_start|>assistant"
) -> str:
    
    if sep_token is not None and sep_token in text:
        t1, t2 = text.split(sep_token, 1)
        if image_placeholder in t1:
            t1 = t1.replace(image_pad, "")
            t1 = t1.replace(image_placeholder, image_pad)
        return t1 + sep_token + t2
    else:
        if image_placeholder in text:
            text = text.replace(image_pad, "")
            text = text.replace(image_placeholder, image_pad)
        return text

def place_output_image(
    text,
    image_pad="<|vision_start|><|image_pad|><|vision_end|>",
    latent_placeholder="<output_image>",
    sep_token="<|im_start|>assistant"
) -> str:
    if latent_placeholder in text:
        text = text.replace(image_pad + '<think>', '<think>')
        text = text.replace(latent_placeholder, image_pad)
    return text

def remove_user_images(examples):
    new_examples = []
    for example in examples:
        new_example = []
        for turn in example:
            new_turn = dict(turn)
            if turn.get("role") == "user":
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            new_example.append(new_turn)
        new_examples.append(new_example)
    return new_examples

def remove_assistant_images(examples):
    new_examples = []
    for example in examples:
        new_example = []
        for turn in example:
            new_turn = dict(turn)
            if turn.get("role") == "assistant":
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            new_example.append(new_turn)
        new_examples.append(new_example)
    return new_examples

def replace_visual_spectial_tokens(texts):
    # 更健壮：找不到分隔符也不崩
    update_texts = []
    for text in texts:
        parts = text.split("<|im_start|>assistant", 1)
        if len(parts) == 2:
            prev, after = parts
            update_texts.append(
                prev + "<|im_start|>assistant" +
                after.replace("<|vision_start|><|image_pad|><|vision_end|>", "<|latent_start|><|image_pad|><|latent_end|>")
            )
        else:
            update_texts.append(
                text.replace("<|vision_start|><|image_pad|><|vision_end|>", "<|latent_start|><|image_pad|><|latent_end|>")
            )
    return update_texts

def replace_subsequent_image_parts_2d(
    input_ids: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    replacement_length: int,
    pad_token: int = 0
) -> torch.Tensor:
    batch_size, _ = input_ids.shape
    replaced_sequences = []
    max_len = 0
    for i in range(batch_size):
        seq_1d = input_ids[i]
        new_seq_1d = replace_subsequent_image_parts_1d(
            seq_1d, start_token, end_token, replacement_token, replacement_length
        )
        replaced_sequences.append(new_seq_1d)
        max_len = max(max_len, new_seq_1d.size(0))
    new_input_ids = input_ids.new_full((batch_size, max_len), fill_value=int(pad_token))
    for i, seq_1d in enumerate(replaced_sequences):
        length = seq_1d.size(0)
        new_input_ids[i, :length] = seq_1d
    return new_input_ids



def replace_subsequent_image_parts_1d(
    seq: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    replacement_length: int
) -> torch.Tensor:
    start_positions = (seq == start_token).nonzero().squeeze(-1)
    end_positions   = (seq == end_token).nonzero().squeeze(-1)
    new_seq_pieces = []
    prev_end = 0
    for (s_pos, e_pos) in zip(start_positions, end_positions):
        new_seq_pieces.append(seq[prev_end : s_pos + 1])
        replacement_span = torch.tensor(
            [replacement_token] * replacement_length,
            dtype=seq.dtype, device=seq.device
        )
        new_seq_pieces.append(replacement_span)
        new_seq_pieces.append(torch.tensor([end_token], dtype=seq.dtype, device=seq.device))
        prev_end = e_pos + 1
    if prev_end < len(seq):
        new_seq_pieces.append(seq[prev_end:])
    new_seq = torch.cat(new_seq_pieces, dim=0)
    return new_seq

def process_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    replacement_length: int,
    pad_token: int = 0
):
    batch_size, _ = input_ids.shape
    processed_sequences = []
    for b in range(batch_size):
        real_tokens = input_ids[b][attention_mask[b] == 1]
        updated_seq = replace_subsequent_image_parts_1d(
            real_tokens, start_token, end_token, replacement_token, replacement_length
        )
        processed_sequences.append(updated_seq)
    new_max_len = max(seq.size(0) for seq in processed_sequences)
    new_input_ids = input_ids.new_full((batch_size, new_max_len), fill_value=int(pad_token))
    new_attention_mask = input_ids.new_zeros((batch_size, new_max_len))
    for b in range(batch_size):
        seq_len_b = processed_sequences[b].size(0)
        new_input_ids[b, :seq_len_b] = processed_sequences[b]
        new_attention_mask[b, :seq_len_b] = 1
    return new_input_ids, new_attention_mask

def find_subsequence(row: torch.Tensor, pattern: torch.Tensor) -> int:
    seq_len = row.size(0)
    pat_len = pattern.size(0)
    for start_idx in range(seq_len - pat_len + 1):
        if torch.all(row[start_idx : start_idx + pat_len] == pattern):
            return start_idx
    return -1

def generate_labels_after_multi_token_start(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    pad_token_idx: int = 0,
    img_token_idx: int = 151655,
) -> torch.Tensor:
    batch_size, _ = input_ids.shape
    labels = input_ids.clone()
    for b in range(batch_size):
        row = labels[b]
        start_idx = find_subsequence(row, start_sequence)
        if start_idx == -1:
            row[:] = -100
        else:
            end_of_subseq = start_idx + start_sequence.size(0)
            row[:end_of_subseq] = -100
        row[row == pad_token_idx] = -100
        row[row == img_token_idx] = -100
    return labels

def generate_labels_after_latent_tokens(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    pad_token_idx: int = 0,
) -> torch.Tensor:
    batch_size, _ = input_ids.shape
    labels = input_ids.clone()
    for b in range(batch_size):
        row = labels[b]
        start_idx = find_subsequence(row, start_sequence)
        if start_idx == -1:
            row[:] = -100
        else:
            row[:start_idx] = -100
        row[row == pad_token_idx] = -100
    return labels

def mask_image_output_tokens(
    input_ids: torch.Tensor,
    image_start_token: int,
    image_token: int
) -> torch.Tensor:
   
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids)
    for i in range(batch_size):
        seq = input_ids[i]
        first_start_pos = -1
        for j in range(seq_len):
            if seq[j] == image_start_token:
                first_start_pos = j
                break
        if first_start_pos == -1:
            continue
        for k in range(first_start_pos + 1, seq_len):
            if seq[k] == image_token:
                mask[i, k] = 1
    return mask

def mask_latent_output_tokens_all_segments(
    input_ids: torch.Tensor,
    latent_start_id: int,
    latent_end_id: int,
    latent_pad_id: int
) -> torch.Tensor:
    B, T = input_ids.shape
    mask = torch.zeros_like(input_ids)
    for b in range(B):
        seq = input_ids[b]
        starts = (seq == latent_start_id).nonzero().squeeze(-1).tolist()
        ends   = (seq == latent_end_id).nonzero().squeeze(-1).tolist()
        e_ptr = 0
        for s in starts:
            while e_ptr < len(ends) and ends[e_ptr] <= s:
                e_ptr += 1
            if e_ptr >= len(ends):
                break
            e = ends[e_ptr]
            if e - s > 1:
                seg = seq[s+1:e]
                mask[b, s+1:e] = (seg == latent_pad_id).to(mask.dtype)
    return mask

def generate_labels_with_latent_template(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    pad_token_idx: int,
    latent_start_id: int,
    latent_end_id: int,
    latent_pad_id: int,
    latent_ce_ratio: float = 1.0,
) -> torch.Tensor:
    B, _ = input_ids.shape
    labels = input_ids.clone()

    def _find_subseq(row: torch.Tensor, pat: torch.Tensor) -> int:
        for i in range(0, row.size(0) - pat.size(0) + 1):
            if torch.all(row[i:i+pat.size(0)] == pat):
                return i
        return -1

    for b in range(B):
        row = labels[b]
        start_idx = _find_subseq(row, start_sequence)
        if start_idx == -1:
            row[:] = -100
            continue
        end_of_subseq = start_idx + start_sequence.size(0)
        row[:end_of_subseq] = -100
        row[row == pad_token_idx] = -100

        seq = input_ids[b]
        start_pos = (seq == latent_start_id).nonzero().squeeze(-1)
        end_pos   = (seq == latent_end_id).nonzero().squeeze(-1)

        in_latent = torch.zeros_like(seq, dtype=torch.bool)
        for s in start_pos.tolist():
            e_candidates = end_pos[end_pos > s]
            if len(e_candidates) == 0:
                continue
            e = int(e_candidates[0].item())
            if e - s > 1:
                in_latent[s+1:e] = True

        outside_latent = ~in_latent
        row[(seq == latent_pad_id) & outside_latent] = -100

        latent_zone = in_latent & (seq == latent_pad_id)
        if latent_ce_ratio < 1.0:
            keep = torch.rand_like(seq.float()) < latent_ce_ratio
            latent_zone = latent_zone & keep.bool()
        drop_zone = (seq == latent_pad_id) & in_latent & ~latent_zone
        row[drop_zone] = -100

    return labels


class LatentTemplateLogitsProcessor(LogitsProcessor):
    def __init__(self, latent_start_id: int, latent_end_id: int, latent_pad_id: int, K: int):
        self.latent_start_id = int(latent_start_id)
        self.latent_end_id   = int(latent_end_id)
        self.latent_pad_id   = int(latent_pad_id)
        self.K               = int(K)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        B = input_ids.size(0)
        for b in range(B):
            seq = input_ids[b]
            start_positions = (seq == self.latent_start_id).nonzero().squeeze(-1)
            end_positions   = (seq == self.latent_end_id).nonzero().squeeze(-1)

            def _last_pos(t):
                return int(t[-1].item()) if t.numel() > 0 else -1

            last_s = _last_pos(start_positions)
            last_e = _last_pos(end_positions)
            inside = (last_s != -1) and (last_e < last_s)

            if not inside:
                scores[b, self.latent_pad_id] = -float("inf")
                scores[b, self.latent_end_id] = -float("inf")
                continue

            i = last_s + 1
            n_pad = 0
            L = seq.size(0)
            while i < L and int(seq[i].item()) == self.latent_pad_id:
                n_pad += 1
                i += 1

            if n_pad < self.K:
                scores[b, :] = -float("inf")
                scores[b, self.latent_pad_id] = 0.0
            else:
                scores[b, :] = -float("inf")
                scores[b, self.latent_end_id] = 0.0
        return scores


if __name__ == "__main__":
    pass
