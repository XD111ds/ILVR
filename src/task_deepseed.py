from PIL import Image
import os

TRAIN_IMAGE_ROOT = "/mnt/public/users/dongshuai/LVR/ILVR_PUB/COMT"
TEST_IMAGE_ROOT  = ""

# ====== Utility Functions ======
def _to_list(x):
    if x is None:
        return []
    return x if isinstance(x, (list, tuple)) else [x]

def _resolve_path(root: str, p: str) -> str:
    if not isinstance(p, str):
        raise ValueError(f"Expected string path, got: {type(p)}")
    return p if os.path.isabs(p) else os.path.join(root, p)

def _assert_exists(p: str):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Image path not found: {p}")


def interleaved_latent_cot_preprocess_function(sample):
    
    # Process user content (Input images)
    user_content = []
    for image_rel in _to_list(sample.get('image_input', [])):
        full_image_path = _resolve_path(TRAIN_IMAGE_ROOT, image_rel)
        _assert_exists(full_image_path)
        user_content.append({"type": "image", "image": full_image_path})
    user_content.append({"type": "text", "text": sample.get("text_input", "")})

    # Process assistant content (Sequence plan with interleaved images/text)
    assistant_content = []
    seq = sample.get('sequence_plan', None)
    if not isinstance(seq, list):
        raise ValueError(f"Missing or non-list 'sequence_plan' field in data sample. Sample keys: {list(sample.keys())}")

    for step in seq:
        stype = step.get('type', None)
        if stype == 'text':
            assistant_content.append({"type": "text", "text": step.get('content', "")})
        elif stype == 'latent':
            helper_rel = step.get('helper_image', None)
            if not helper_rel:
                raise ValueError(f"Latent step is missing 'helper_image': {step}")
            helper_path = _resolve_path(TRAIN_IMAGE_ROOT, helper_rel)
            _assert_exists(helper_path)
            assistant_content.append({"type": "image", "image": helper_path})
        else:
            raise ValueError(f"Unknown step type: {stype}")

    conversations = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    return conversations


def single_input_image_preprocess_function(sample):
    
    inp_path  = sample.get("image_input")
    out_path  = sample.get("image_output")
    if inp_path is None or out_path is None:
        raise ValueError("Sample is missing 'image_input' or 'image_output'")

    inp_full = _resolve_path(TRAIN_IMAGE_ROOT, inp_path) if not os.path.isabs(inp_path) else inp_path
    out_full = _resolve_path(TRAIN_IMAGE_ROOT, out_path) if not os.path.isabs(out_path) else out_path
    _assert_exists(inp_full)
    _assert_exists(out_full)

    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": inp_full},
                {"type": "text",  "text":  sample.get("text_input", "")},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "image", "image": out_full},
                {"type": "text",  "text":  sample.get("text_output", "")},
            ],
        }
    ]
    return conversations


def single_input_image_test_preprocess_function(sample):
    inp_path  = sample.get("image_input")
    if inp_path is None:
        raise ValueError("Sample is missing 'image_input'")

    inp_full = _resolve_path(TEST_IMAGE_ROOT, inp_path) if not os.path.isabs(inp_path) else inp_path
    _assert_exists(inp_full)

    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": inp_full},
                {"type": "text",  "text":  sample.get("text_input", "")},
            ],
        },
    ]
    return conversations


def multiple_input_images_preprocess_function(sample):
    user_content = []
    for rel in _to_list(sample.get('image_input', [])):
        full = _resolve_path(TRAIN_IMAGE_ROOT, rel) if not os.path.isabs(rel) else rel
        _assert_exists(full)
        user_content.append({"type": "image", "image": full})
    user_content.append({"type": "text", "text": sample.get("text_input", "")})

    out_path = sample.get("image_output")
    if out_path is None:
        raise ValueError("Sample is missing 'image_output'")
    out_full = _resolve_path(TRAIN_IMAGE_ROOT, out_path) if not os.path.isabs(out_path) else out_path
    _assert_exists(out_full)

    conversations = [
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": [
                {"type": "image", "image": out_full},
                {"type": "text",  "text":  sample.get("text_output", "")},
            ],
        },
    ]
    return conversations

def multiple_input_images_test_preprocess_function(sample):
    user_content = []
    for rel in _to_list(sample.get('image_input', [])):
        full = _resolve_path(TEST_IMAGE_ROOT, rel) if not os.path.isabs(rel) else rel
        _assert_exists(full)
        user_content.append({"type": "image", "image": full})
    user_content.append({"type": "text", "text": sample.get("text_input", "")})

    conversations = [{"role": "user", "content": user_content}]
    return conversations


def interleaved_latent_cot_test_preprocess_function(sample):
    user_content = []
    for rel in _to_list(sample.get('image_input', [])):
        full = _resolve_path(TEST_IMAGE_ROOT, rel) if not os.path.isabs(rel) else rel
        _assert_exists(full)
        user_content.append({"type": "image", "image": full})
    user_content.append({"type": "text", "text": sample.get("text_input", "")})
    conversations = [{"role": "user", "content": user_content}]
    return conversations

task_preporcess_config = {
    'vsp-spatial-reasoning': single_input_image_preprocess_function,
    'vsp-spatial-planning': single_input_image_preprocess_function,
    'blink-jigsaw': multiple_input_images_preprocess_function,
    'sat': multiple_input_images_preprocess_function,
    'vsp-spatial-planning-cot': interleaved_latent_cot_preprocess_function,
    'zebra-cot': interleaved_latent_cot_preprocess_function,
}

task_test_preporcess_config = {
    'vsp-spatial-reasoning': single_input_image_test_preprocess_function,
    'vsp-spatial-planning': single_input_image_test_preprocess_function,
    'blink-jigsaw': multiple_input_images_test_preprocess_function,
    'sat': multiple_input_images_test_preprocess_function,
    'vsp-spatial-planning-cot': interleaved_latent_cot_test_preprocess_function,
    'zebra-cot': interleaved_latent_cot_test_preprocess_function,
}


ACTION_MAP = {
    "LEFT":  (0, -1),
    "DOWN":  (1,  0),
    "RIGHT": (0,  1),
    "UP":    (-1, 0),
    "L":  (0, -1),
    "D":  (1,  0),
    "R":  (0,  1),
    "U":  (-1, 0)
}

def parse_action_sequence(path_str):
    path_str = (path_str or "").upper()
    return [ch for ch in list(path_str) if ch in ['U', 'D', 'R', 'L']]

def simulate_vsp(map_desc, path_str):
    action_sequence = parse_action_sequence(path_str)

    start = None
    for r, row in enumerate(map_desc):
        for c, val in enumerate(row):
            if val == 1:
                start = (r, c)
                break
        if start is not None:
            break

    if start is None:
        raise ValueError("The map description does not contain a start position (cell value 1).")

    current_position = start
    for action in action_sequence:
        if action not in ACTION_MAP:
            return {"success": False, "status": "Invalid action", "invalid": True}

        dr, dc = ACTION_MAP[action]
        new_r = current_position[0] + dr
        new_c = current_position[1] + dc

        if new_r < 0 or new_r >= len(map_desc) or new_c < 0 or new_c >= len(map_desc[0]):
            continue

        current_position = (new_r, new_c)

        if map_desc[new_r][new_c] == -1:
            return {"success": False, "status": "Fell in hole", "invalid": False}

    final_r, final_c = current_position
    if map_desc[final_r][final_c] == 2:
        return {"success": True, "status": "Reached goal", "invalid": False}
    else:
        return {"success": False, "status": "Did not reach goal", "invalid": False}