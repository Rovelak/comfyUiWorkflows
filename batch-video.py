import os
import json
import time
import uuid
import shutil
import random
from pathlib import Path

import requests

# =========================
# CONFIG
# =========================

COMFYUI_URL = "http://127.0.0.1:8188"

COMFYUI_DIR = Path(r"D:\Home\ComfyUI_windows_portable\ComfyUI")

WORKFLOW_PATH = Path(r"D:\Home\ComfyUI_windows_portable\ComfyUI\user\default\workflows\ltx-2.3 combi 1.1-wow.stable-fast.noaudio-export.json")

INPUT_IMAGES_DIR = Path(r"D:\Home\images_a_animer")

FINAL_OUTPUT_DIR = Path(r"D:\Home\final_videos")

POSITIVE_PROMPT = (
    "Animate this single still image into one continuous shot. Preserve the exact same scene, same people, same clothing, same environment, same composition, and same camera angle."
)

PROMPT_VARIANTS = [
    "Add subtle natural motion with a gentle camera push-in.",
    "Add soft parallax and delicate environmental motion.",
    "Add subtle cinematic movement with very light depth motion.",
    "Add gentle natural movement and a calm atmospheric feel.",
]

NEGATIVE_PROMPT = (
    "scene change, cut, transition, different shot, different location, different person, extra person, subject replacement, face change, body change, clothing change, gender swap, age change, warping, morphing, hallucinated background, new objects, new characters, strong motion, camera shake, flicker, jitter, blur, low quality, watermark, subtitles, overlay, text"
)

VIDEO_SECONDS = 2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# =========================
# HELPERS
# =========================

def ensure_dirs():
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (COMFYUI_DIR / "input").mkdir(parents=True, exist_ok=True)
    (COMFYUI_DIR / "output").mkdir(parents=True, exist_ok=True)

def load_workflow() -> dict:
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def set_seed_if_present(wf: dict, seed: int):
    candidates = [
        ("115", "noise_seed"),
        ("209:115", "noise_seed"),
        ("232", "seed"),
        ("271", "seed"),
    ]

    for node, key in candidates:
        if node in wf and "inputs" in wf[node] and key in wf[node]["inputs"]:
            wf[node]["inputs"][key] = seed

def set_strength_if_present(wf: dict, strength: float):
    nodes = ["209:154", "209:213"]

    for node in nodes:
        if node in wf and "inputs" in wf[node] and "strength" in wf[node]["inputs"]:
            wf[node]["inputs"]["strength"] = strength

def prepare_workflow_for_image(workflow, image_filename, output_prefix, seed, strength, prompt):

    wf = json.loads(json.dumps(workflow))

    wf["149"]["inputs"]["image"] = image_filename

    wf["121"]["inputs"]["text"] = prompt
    wf["593"]["inputs"]["text"] = NEGATIVE_PROMPT

    wf["196"]["inputs"]["Xi"] = VIDEO_SECONDS
    wf["196"]["inputs"]["Xf"] = VIDEO_SECONDS

    wf["188"]["inputs"]["filename_prefix"] = output_prefix

    set_seed_if_present(wf, seed)
    set_strength_if_present(wf, strength)

    return wf

def queue_prompt(prompt_workflow: dict) -> str:
    client_id = str(uuid.uuid4())

    payload = {
        "prompt": prompt_workflow,
        "client_id": client_id,
    }

    r = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=60)
    r.raise_for_status()

    return r.json()["prompt_id"]

def wait_for_completion(prompt_id: str):

    while True:

        r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        r.raise_for_status()

        history = r.json()

        if prompt_id in history:
            return history[prompt_id]

        time.sleep(2)

def extract_video_files_from_history(history_entry: dict):
    results = []

    outputs = history_entry.get("outputs", {})

    for node_id, node_output in outputs.items():
        for key in ("gifs", "images", "files"):
            if key in node_output:
                for item in node_output[key]:
                    filename = item.get("filename")
                    subfolder = item.get("subfolder", "")
                    filetype = item.get("type", "output")

                    if filename:
                        results.append({
                            "node_id": node_id,
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": filetype,
                        })

    return results

def resolve_output_path(file_info):
    filetype = file_info.get("type", "output")
    filename = file_info["filename"]
    subfolder = file_info.get("subfolder", "")

    if filetype == "temp":
        base = COMFYUI_DIR / "temp"
    else:
        base = COMFYUI_DIR / "output"

    return base / subfolder / filename
    
def find_best_video_output(files: list):
    if not files:
        return None

    # priorité au mp4
    mp4_files = [f for f in files if f["filename"].lower().endswith(".mp4")]
    if mp4_files:
        return mp4_files[0]

    # sinon gif/webm/mov
    video_like = [
        f for f in files
        if f["filename"].lower().endswith((".webm", ".mov", ".mkv", ".avi", ".gif"))
    ]
    if video_like:
        return video_like[0]

    return files[0]

def copy_image_to_comfy_input(image_path):

    target = COMFYUI_DIR / "input" / image_path.name

    if target.exists():

        new_name = f"{image_path.stem}_{uuid.uuid4().hex[:8]}{image_path.suffix}"
        target = COMFYUI_DIR / "input" / new_name

    shutil.copy2(image_path, target)

    return target.name

def process_image(image_path):

    print(f"\n=== Processing: {image_path.name} ===")

    comfy_name = copy_image_to_comfy_input(image_path)

    base_name = image_path.stem

    output_prefix = f"batch_ltx23/{base_name}"

    seed = random.randint(1, 10_000_000)
    strength = round(random.uniform(0.52, 0.68), 3)

    prompt = POSITIVE_PROMPT + " " + random.choice(PROMPT_VARIANTS)

    print(f"Seed: {seed} | Strength: {strength}")

    workflow = load_workflow()

    workflow = prepare_workflow_for_image(
        workflow,
        comfy_name,
        output_prefix,
        seed,
        strength,
        prompt
    )

    prompt_id = queue_prompt(workflow)

    history = wait_for_completion(prompt_id)

    files = extract_video_files_from_history(history)

    if not files:
        raise RuntimeError("No output found in ComfyUI history")

    best_file = find_best_video_output(files)
    if not best_file:
        raise RuntimeError("No usable output file found")

    video_file = resolve_output_path(best_file)

    print("History outputs found:")
    for f in files:
        print(f"  - node={f.get('node_id')} type={f.get('type')} subfolder={f.get('subfolder')} file={f.get('filename')}")

    print(f"Selected output: {video_file}")

    if not video_file.exists():
        raise FileNotFoundError(f"Generated file not found: {video_file}")

    final_ext = video_file.suffix if video_file.suffix else ".mp4"
    final_path = FINAL_OUTPUT_DIR / f"{base_name}{final_ext}"

    shutil.copy2(video_file, final_path)

    print(f"Saved: {final_path}")

def main():

    ensure_dirs()

    images = sorted(
        p for p in INPUT_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Found {len(images)} images")

    for image in images:

        try:
            process_image(image)
        except Exception as e:
            print(f"[ERROR] {image.name}: {e}")

    print("\nDone")

if __name__ == "__main__":
    main()