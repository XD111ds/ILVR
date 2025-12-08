# evaluate_deepseed.py
import os
import re
import json
import argparse
import logging
from typing import List, Dict, Any, Union

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, LogitsProcessorList

from utils_deepseed import LatentTemplateLogitsProcessor  

try:
    from mathruler.grader import extract_boxed_content
except Exception:
    extract_boxed_content = None
import re

ACTION_MAP = {
    "LEFT": (0,-1), "DOWN": (1,0), "RIGHT": (0,1), "UP": (-1,0),
    "L": (0,-1), "D": (1,0), "R": (0,1), "U": (-1,0),
}


BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  
BASE_DATASET_DIR = '/mnt/public/users/dongshuai/LVR/ILVR_PUB/COMT'


def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data jsonl file.")
    parser.add_argument("--task_name", type=str, required=True, help="Task name for bookkeeping.")
    parser.add_argument("--output_json_path", type=str, default="evaluation_results.jsonl", help="Where to save per-sample results.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="HF cache dir.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    return parser.parse_args()


def load_processor(model_dir: str, cache_dir: str):

    try:
        return AutoProcessor.from_pretrained(model_dir, cache_dir=cache_dir)
    except Exception:
        logging.warning("Failed to load processor from model_dir, falling back to base model id.")
        return AutoProcessor.from_pretrained(BASE_MODEL_ID, cache_dir=cache_dir)

def load_model(model_dir: str, cache_dir: str):

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2"
        )
    except Exception as e:
        logging.warning(f"flash_attention_2 initialization failed ({e}), falling back to eager.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation="eager"
        )
    model.eval()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    return model

def _resolve_image_paths(image_input: Union[str, List[str]]) -> List[str]:

    if image_input is None:
        return []
    if isinstance(image_input, str):
        p = image_input
        if not os.path.isabs(p):
            p = os.path.join(BASE_DATASET_DIR, p)
        return [p]
    if isinstance(image_input, list):
        outs = []
        for p in image_input:
            if not os.path.isabs(p):
                p = os.path.join(BASE_DATASET_DIR, p)
            outs.append(p)
        return outs
    return []

def open_images(paths: List[str]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        # Raise error immediately for directories, be explicit
        if os.path.isdir(p):
            raise IsADirectoryError(f"Expected image file, but received directory path: {p}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


# ---- Several fallback rules for parsing the final answer ----
_yes_set = {"yes", "true", "a"}
_no_set  = {"no", "false", "b"}

def extract_final_answer(text: str) -> str:

    s = text.strip()
    patterns = [
        r"final\s*answer\s*(?:is)?\s*[:：]\s*(.+)",
        r"answer\s*(?:is)?\s*[:：]\s*(.+)"
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = re.split(r"[\n\r]", cand)[0]
            return cand
    tokens = re.findall(r"[A-Za-z]+", s.lower())
    only = [t for t in tokens if t in (_yes_set | _no_set)]
    if len(only) == 1:
        return only[0].capitalize() if only[0] in {"yes", "no"} else only[0]
    return s

def _strip_special_tokens(s: str) -> str:
    # Common end/special tokens, extend if necessary
    end_markers = [
        "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>",
        "</s>", "<s>", "[/INST]", "[INST]"
    ]
    for m in end_markers:
        s = s.replace(m, "")
    return s.strip()

def extract_assistant_content(text: str) -> str:

    s = text or ""


    m = re.search(r"<\|im_start\|>\s*assistant\s*(.*?)(?:<\|im_end\|>|$)", s, flags=re.S|re.I)
    if m:
        return _strip_special_tokens(m.group(1))


    m = re.search(
        r"<\|assistant\|>\s*(.*?)(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|<\|end\|>|$)",
        s, flags=re.S|re.I
    )
    if m:
        return _strip_special_tokens(m.group(1))


    m = re.search(r"(?:^|\n|\r)assistant\s*[:：]?\s*(.*)$", s, flags=re.S|re.I)
    if m:
        return _strip_special_tokens(m.group(1))


    return _strip_special_tokens(s)


def normalize_for_match(ans: str) -> str:
    a = ans.strip()
    low = a.lower()
    if low in _yes_set:
        return "Yes"
    if low in _no_set:
        return "No"
    return a

def run_one_example(model, processor, sample: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    text_input = sample.get("text_input", "")
    img_list = _resolve_image_paths(sample.get("image_input", []))
    images = open_images(img_list) if img_list else None


    content = []
    if images:
        for _ in images:
            content.append({"type": "image"})
    content.append({"type": "text", "text": text_input})
    conversations = [
        {"role": "user",   "content": content}
    ]

    prompt = processor.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=images,   
        return_tensors="pt",
        padding=True
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    latent_pad_id   = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_start_id = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_end_id   = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0,0].item()
    K = int(getattr(model.config, "latent_size", 8))
    lp = LogitsProcessorList([
        LatentTemplateLogitsProcessor(latent_start_id, latent_end_id, latent_pad_id, K)
    ])

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            **gen_kwargs,
            tokenizer=processor.tokenizer,
        )
    out_text = processor.batch_decode(out_ids, skip_special_tokens=False)[0].strip()

    extracted = extract_final_answer(out_text)
    return {"raw_output": out_text, "extracted_final_answer": extracted}

def main():
    args = get_eval_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info(f"Loading model: {args.model_dir}")
    model = load_model(args.model_dir, args.cache_dir)
    processor = load_processor(args.model_dir, args.cache_dir)


    data = []
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if not data:
        logging.error("Test set is empty, exiting.")
        return

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(args.temperature > 0)
    )

    total = len(data)
    success = 0
    results = []

    logging.info(f"Starting evaluation: num_samples={total}")
    for i, sample in enumerate(data):
        try:
            pred = run_one_example(model, processor, sample, gen_kwargs)
            out_text = pred["raw_output"]
            assistant_only = extract_assistant_content(out_text)

            if args.task_name == "vsp-spatial-planning-cot":
                
                map_desc = sample.get("map_desc", [])
                path_str = extract_path_from_text(out_text)
                sim = simulate_vsp(map_desc, path_str)
                ok = bool(sim["success"])
                if ok: success += 1

                results.append({
                    "index": i,
                    "task_name": args.task_name,
                    "text_input": sample.get("text_input", ""),
                    "image_input": sample.get("image_input", []),
                    "predicted_full_output": out_text,
                    "extracted_path_string": path_str,
                    "vsp_simulation_result": sim,
                    "match": ok,
                })

            else:  # Standard tasks
                gold = sample.get("original_final_answer", "")
                pred_norm = normalize_for_match(pred["extracted_final_answer"])
                gold_norm = normalize_for_match(gold)
                ok = (pred_norm == gold_norm)
                if ok: success += 1
                
                results.append({
                    "index": i,
                    "task_name": args.task_name,
                    
                    "prediction": assistant_only,
                    "gold_final_answer": gold,
                    "match": bool(ok)
                })

            if (i + 1) % 10 == 0 or (i + 1) == total:
                logging.info(f"[{i+1}/{total}] Current accuracy={success/(i+1):.4f}")

        except Exception as e:
            logging.error(f"[{i}] Evaluation failed: {e}")
            results.append({
                "index": i,
                "task_name": args.task_name,
                "error": str(e)
            })

    acc = success / total
    with open(args.output_json_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    logging.info(f"Evaluation finished: success {success}/{total}; accuracy={acc:.4f}")
    print(f"Accuracy = {acc:.4f}  ({success}/{total})")
    



def parse_action_sequence(path_str: str):
    s = (path_str or "").upper()
    return [ch for ch in s if ch in ['U','D','R','L']]

def simulate_vsp(map_desc, path_str):
    actions = parse_action_sequence(path_str)

    # Find start point 1
    start = None
    for r, row in enumerate(map_desc):
        for c, val in enumerate(row):
            if val == 1:
                start = (r, c); break
        if start is not None: break
    if start is None:
        raise ValueError("The map description does not contain a start position (cell value 1).")

    cur = start
    for a in actions:
        if a not in ACTION_MAP:
            return {"success": False, "status": "Invalid action", "invalid": True}
        dr, dc = ACTION_MAP[a]
        nr, nc = cur[0] + dr, cur[1] + dc
        if nr < 0 or nr >= len(map_desc) or nc < 0 or nc >= len(map_desc[0]):
            continue
        cur = (nr, nc)
        if map_desc[nr][nc] == -1:
            return {"success": False, "status": "Fell in hole", "invalid": False}

    return {"success": map_desc[cur[0]][cur[1]] == 2,
            "status": "Reached goal" if map_desc[cur[0]][cur[1]] == 2 else "Did not reach goal",
            "invalid": False}

def extract_path_from_text(generated_text: str) -> str:
    if extract_boxed_content is not None:
        s = extract_boxed_content(generated_text)
        if s: return s
    m = re.search(r"\\boxed\{([UDLRudlr]+)\}", generated_text)
    if m: return m.group(1)
    m = re.search(r"final\s*answer\s*(?:is)?\s*[:：]\s*([UDLRudlr]+)", generated_text, re.I)
    return m.group(1) if m else ""


if __name__ == "__main__":
    main()