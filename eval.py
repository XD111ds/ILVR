# evaluate_deepseed.py
import os
import re
import json
import argparse
import logging
import time
from typing import List, Dict, Any, Union

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

try:
    from mathruler.grader import extract_boxed_content
except Exception:
    extract_boxed_content = None


ACTION_MAP = {
    "LEFT": (0, -1), "DOWN": (1, 0), "RIGHT": (0, 1), "UP": (-1, 0),
    "L": (0, -1), "D": (1, 0), "R": (0, 1), "U": (-1, 0),
}

BASE_MODEL_ID = ""  

BASE_DATASET_DIR = ''



def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the trained model directory.")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to the test data JSONL file.")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Task name (used for bookkeeping and branching logic).")
    parser.add_argument("--output_json_path", type=str, default="evaluation_results.jsonl",
                        help="Path to save per-sample evaluation results.")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Hugging Face cache directory.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    return parser.parse_args()


def load_processor(model_dir: str, cache_dir: str):
    """
    Prefer loading the processor from the finetuned model directory
    (including any special tokens). If this fails, fall back to the
    base model processor.

    During evaluation, no new special tokens should be introduced,
    as this may cause mismatches with the model weights.
    """
    try:
        return AutoProcessor.from_pretrained(model_dir, cache_dir=cache_dir)
    except Exception:
        logging.warning(
            "Failed to load processor from model_dir; falling back to the base model."
        )
        return AutoProcessor.from_pretrained(BASE_MODEL_ID, cache_dir=cache_dir)


def load_model(model_dir: str, cache_dir: str):
    """
    Prefer flash_attention_2 for efficiency; automatically fall back
    to eager attention if flash attention is unavailable.
    """
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        logging.warning(
            f"flash_attention_2 initialization failed ({e}); "
            f"falling back to eager attention."
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation="eager",
        )

    model.eval()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    return model


def _resolve_image_paths(image_input: Union[str, List[str]]) -> List[str]:
    """
    Resolve image paths from either a single string or a list of strings.

    - Relative paths are resolved using BASE_DATASET_DIR.
    - Absolute paths are returned unchanged.
    """
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
    """
    Load images from disk and convert them to RGB.

    Raises explicit errors if a directory path or a non-existent file is encountered.
    """
    imgs = []
    for p in paths:
        if os.path.isdir(p):
            raise IsADirectoryError(
                f"Expected an image file, but received a directory path: {p}"
            )
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


# ---- Fallback rules for parsing the final answer ----
_yes_set = {"yes", "true", "a"}
_no_set = {"no", "false", "b"}


def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from the model output and return a simplified string.

    Priority rules:
      1) Match 'final answer is: XXX' or 'final answer: XXX' (case-insensitive)
      2) Match 'answer is: XXX' or 'answer: XXX'
      3) If the output contains exactly one of yes/no/true/false/a/b
         (case-insensitive), return it directly
      4) Otherwise, fall back to the full text (useful for debugging)
    """
    s = text.strip()
    patterns = [
        r"final\s*answer\s*(?:is)?\s*[:：]\s*(.+)",
        r"answer\s*(?:is)?\s*[:：]\s*(.+)",
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
    """
    Remove common special or end-of-sequence markers from decoded text.
    """
    end_markers = [
        "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>",
        "</s>", "<s>", "[/INST]", "[INST]",
    ]
    for m in end_markers:
        s = s.replace(m, "")
    return s.strip()


def extract_assistant_content(text: str) -> str:
    """
    Extract only the assistant's content from the raw model output.

    Designed to be compatible with ChatML / Qwen / LLaMA-style templates.

    Priority:
      1) <|im_start|>assistant ... <|im_end|>
      2) <|assistant|> ... (until an end marker or end of text)
      3) Fallback: content after the last occurrence of 'assistant'
    """
    s = text or ""

    # 1) ChatML-style pattern
    m = re.search(
        r"<\|im_start\|>\s*assistant\s*(.*?)(?:<\|im_end\|>|$)",
        s,
        flags=re.S | re.I,
    )
    if m:
        return _strip_special_tokens(m.group(1))

    # 2) Qwen / other styles
    m = re.search(
        r"<\|assistant\|>\s*(.*?)(?:<\|im_end\|>|<\|endoftext\|>|"
        r"<\|eot_id\|>|<\|end\|>|$)",
        s,
        flags=re.S | re.I,
    )
    if m:
        return _strip_special_tokens(m.group(1))

    # 3) Fallback: last assistant segment
    m = re.search(
        r"(?:^|\n|\r)assistant\s*[:：]?\s*(.*)$",
        s,
        flags=re.S | re.I,
    )
    if m:
        return _strip_special_tokens(m.group(1))

    # If nothing matches, return cleaned full text
    return _strip_special_tokens(s)


def normalize_for_match(ans: str) -> str:
    """
    Normalize answers for matching (mainly for yes/no style tasks).
    """
    a = ans.strip()
    low = a.lower()
    if low in _yes_set:
        return "Yes"
    if low in _no_set:
        return "No"
    return a


def run_one_example(
    model,
    processor,
    sample: Dict[str, Any],
    gen_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run inference on a single example and return raw output,
    extracted final answer, and inference time.
    """
    text_input = sample.get("text_input", "")
    img_list = _resolve_image_paths(sample.get("image_input", []))
    images = open_images(img_list) if img_list else None

    # Align with training: put image entries into user.content, then append text
    content = []
    if images:
        for _ in images:
            content.append({"type": "image"})
    content.append({"type": "text", "text": text_input})

    conversations = [
        {"role": "user", "content": content}
    ]

    prompt = processor.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True,
    )

    # Move tensor inputs to the model device
    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }


    # ---- Start inference timing (generation only) ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            **gen_kwargs,
            tokenizer=processor.tokenizer,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    # ---- End inference timing ----

    out_text = processor.batch_decode(
        out_ids, skip_special_tokens=False
    )[0].strip()

    extracted = extract_final_answer(out_text)
    return {
        "raw_output": out_text,
        "extracted_final_answer": extracted,
        "inference_time": inference_time,
    }


def main():
    args = get_eval_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Loading model from: {args.model_dir}")
    model = load_model(args.model_dir, args.cache_dir)
    processor = load_processor(args.model_dir, args.cache_dir)

    # Load test data
    data = []
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if not data:
        logging.error("Test dataset is empty. Exiting.")
        return

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(args.temperature > 0),
    )

    total = len(data)
    success = 0
    results = []
    total_inference_time = 0.0

    logging.info(f"Starting evaluation: num_samples={total}")
    for i, sample in enumerate(data):
        try:
            pred = run_one_example(model, processor, sample, gen_kwargs)
            sample_infer_time = pred.get("inference_time", None)
            if sample_infer_time is not None:
                total_inference_time += sample_infer_time

            out_text = pred["raw_output"]
            assistant_only = extract_assistant_content(out_text)

            if args.task_name == "vsp-spatial-planning-cot":
                # Extract path and run simulation
                map_desc = sample.get("map_desc", [])
                path_str = extract_path_from_text(out_text)
                sim = simulate_vsp(map_desc, path_str)
                ok = bool(sim["success"])
                if ok:
                    success += 1

                results.append({
                    "index": i,
                    "task_name": args.task_name,
                    "text_input": sample.get("text_input", ""),
                    "image_input": sample.get("image_input", []),
                    "predicted_full_output": out_text,
                    "extracted_path_string": path_str,
                    "vsp_simulation_result": sim,
                    "match": ok,
                    "inference_time_sec": sample_infer_time,
                })

            else:
                # Default zebra-cot branch
                gold = sample.get("original_final_answer", "")
                pred_norm = normalize_for_match(pred["extracted_final_answer"])
                gold_norm = normalize_for_match(gold)
                ok = (pred_norm == gold_norm)
                if ok:
                    success += 1

                results.append({
                    "index": i,
                    "task_name": args.task_name,
                    "image_input": sample.get("image_input", []),
                    "prediction": assistant_only,
                    "gold_final_answer": gold,
                    "match": bool(ok),
                    "inference_time_sec": sample_infer_time,
                })

            if (i + 1) % 10 == 0 or (i + 1) == total:
                logging.info(
                    f"[{i+1}/{total}] Current accuracy = {success/(i+1):.4f}"
                )

        except Exception as e:
            logging.error(f"[{i}] Evaluation failed: {e}")
            results.append({
                "index": i,
                "task_name": args.task_name,
                "error": str(e),
                "inference_time_sec": None,
            })

    acc = success / total if total > 0 else 0.0
    avg_inference_time = (
        total_inference_time / total if total > 0 else 0.0
    )

    with open(args.output_json_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Write a summary line as the last entry
        fout.write(json.dumps({
            "summary": True,
            "num_samples": total,
            "average_inference_time_sec": avg_inference_time,
            "total_inference_time_sec": total_inference_time,
            "accuracy": acc,
            "success": success,
        }, ensure_ascii=False) + "\n")

    logging.info(
        f"Evaluation finished: {success}/{total} correct; accuracy={acc:.4f}"
    )
    print(f"Accuracy = {acc:.4f} ({success}/{total})")


def parse_action_sequence(path_str: str):
    """
    Parse a path string into a sequence of movement actions.
    """
    s = (path_str or "").upper()
    return [ch for ch in s if ch in ['U', 'D', 'R', 'L']]


def simulate_vsp(map_desc, path_str):
    """
    Simulate the spatial planning environment given a map and an action sequence.
    """
    actions = parse_action_sequence(path_str)

    # Locate the start position (cell value = 1)
    start = None
    for r, row in enumerate(map_desc):
        for c, val in enumerate(row):
            if val == 1:
                start = (r, c)
                break
        if start is not None:
            break

    if start is None:
        raise ValueError(
            "The map description does not contain a start position (cell value 1)."
        )

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
            return {"success": False, "status": "Fell into a hole", "invalid": False}

    return {
        "success": map_desc[cur[0]][cur[1]] == 2,
        "status": (
            "Reached goal" if map_desc[cur[0]][cur[1]] == 2
            else "Did not reach goal"
        ),
        "invalid": False,
    }


def extract_path_from_text(generated_text: str) -> str:
    """
    Extract a movement path string from model output.

    Priority:
      1) Use mathruler's boxed-content extractor if available
      2) Match '\\boxed{UDLR...}'
      3) Match 'final answer is: UDLR...'
    """
    if extract_boxed_content is not None:
        s = extract_boxed_content(generated_text)
        if s:
            return s

    m = re.search(r"\\boxed\{([UDLRudlr]+)\}", generated_text)
    if m:
        return m.group(1)

    m = re.search(
        r"final\s*answer\s*(?:is)?\s*[:：]\s*([UDLRudlr]+)",
        generated_text,
        re.I,
    )
    return m.group(1) if m else ""


if __name__ == "__main__":
    main()
