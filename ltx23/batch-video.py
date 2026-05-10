import os
import json
import time
import uuid
import shutil
import random
import csv
import subprocess
from pathlib import Path

import requests

# =========================
# CONFIG
# =========================

COMFYUI_URL = "http://127.0.0.1:8188"

COMFYUI_DIR = Path(r"D:\IA\ComfyUI_windows_portable\ComfyUI")

WORKFLOW_PATH = Path(r"D:\IA\comfyUiWorkflows\ltx23\perso.json")

INPUT_IMAGES_DIR = Path(r"D:\Home\images_a_animer")

FINAL_OUTPUT_DIR = Path(r"D:\Home\final_videos")

PERF_LOG_PATH = FINAL_OUTPUT_DIR / "batch_ltx23_perf.csv"

# Slower per image, but useful if ComfyUI becomes inconsistent over long runs.
# It asks ComfyUI to unload cached models after each completed prompt.
FREE_COMFY_MEMORY_AFTER_EACH_IMAGE = False

GPU_MONITOR_INTERVAL_S = 10

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

VARIANTS_PER_IMAGE = 4

MAX_IMAGES = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}

# =========================
# HELPERS
# =========================

def ensure_dirs():
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (COMFYUI_DIR / "input").mkdir(parents=True, exist_ok=True)
    (COMFYUI_DIR / "output").mkdir(parents=True, exist_ok=True)

def get_gpu_snapshot() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,pstate",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception as e:
        return {"gpu_error": str(e)}

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return {"gpu_error": line}

    return {
        "gpu_memory_used_mb": parts[0],
        "gpu_memory_total_mb": parts[1],
        "gpu_util_percent": parts[2],
        "gpu_temp_c": parts[3],
        "gpu_pstate": parts[4],
    }

def to_int(value, default=0):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default

def summarize_gpu_samples(samples: list) -> dict:
    usable_samples = [s for s in samples if "gpu_error" not in s]
    if not usable_samples:
        return {
            "gpu_samples": len(samples),
            "gpu_memory_peak_mb": "",
            "gpu_util_peak_percent": "",
            "gpu_util_avg_percent": "",
            "gpu_temp_peak_c": "",
        }

    memory_values = [to_int(s.get("gpu_memory_used_mb")) for s in usable_samples]
    util_values = [to_int(s.get("gpu_util_percent")) for s in usable_samples]
    temp_values = [to_int(s.get("gpu_temp_c")) for s in usable_samples]

    return {
        "gpu_samples": len(usable_samples),
        "gpu_memory_peak_mb": max(memory_values),
        "gpu_util_peak_percent": max(util_values),
        "gpu_util_avg_percent": round(sum(util_values) / len(util_values), 1),
        "gpu_temp_peak_c": max(temp_values),
    }

def get_gpu_process_snapshot() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception as e:
        return f"gpu_process_error={e}"

    return " | ".join(line.strip() for line in result.stdout.splitlines() if line.strip())

def append_perf_log(row: dict):
    PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "image",
        "variant",
        "seconds",
        "seed",
        "strength",
        "gpu_before",
        "gpu_after",
        "gpu_samples",
        "gpu_memory_peak_mb",
        "gpu_util_peak_percent",
        "gpu_util_avg_percent",
        "gpu_temp_peak_c",
        "gpu_processes_before",
        "gpu_processes_after",
        "status",
        "error",
    ]
    exists = PERF_LOG_PATH.exists()

    with open(PERF_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})

def free_comfy_memory():
    payload = {"unload_models": True, "free_memory": True}
    r = requests.post(f"{COMFYUI_URL}/free", json=payload, timeout=60)
    r.raise_for_status()

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

def wait_for_completion(prompt_id: str, gpu_samples=None):

    last_gpu_sample_at = 0

    while True:

        r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        r.raise_for_status()

        history = r.json()

        if prompt_id in history:
            return history[prompt_id]

        now = time.monotonic()
        if gpu_samples is not None and now - last_gpu_sample_at >= GPU_MONITOR_INTERVAL_S:
            gpu_samples.append(get_gpu_snapshot())
            last_gpu_sample_at = now

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

def get_input_images():
    images = sorted(
        p for p in INPUT_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]

    return images

def expected_variant_output(base_name: str, variant_index: int, extension: str = ".mp4") -> Path:
    return FINAL_OUTPUT_DIR / f"{base_name}_ltx23_v{variant_index:02d}{extension}"

def find_existing_variant_output(base_name: str, variant_index: int):
    expected_mp4 = expected_variant_output(base_name, variant_index)
    if expected_mp4.exists():
        return expected_mp4

    prefix = f"{base_name}_ltx23_v{variant_index:02d}"
    for path in sorted(FINAL_OUTPUT_DIR.glob(f"{prefix}.*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            return path

    return None

def process_variant(image_path, comfy_name, workflow_template, variant_index):

    started_at = time.monotonic()
    gpu_before = get_gpu_snapshot()
    gpu_processes_before = get_gpu_process_snapshot()

    base_name = image_path.stem

    output_stem = f"{base_name}_ltx23_v{variant_index:02d}"
    output_prefix = f"batch_ltx23/{output_stem}"

    seed = random.randint(1, 10_000_000)
    strength = round(random.uniform(0.52, 0.68), 3)

    prompt = POSITIVE_PROMPT + " " + random.choice(PROMPT_VARIANTS)

    print(f"  variant {variant_index}: seed={seed} | strength={strength}")

    workflow = prepare_workflow_for_image(
        workflow_template,
        comfy_name,
        output_prefix,
        seed,
        strength,
        prompt
    )

    try:
        prompt_id = queue_prompt(workflow)

        gpu_samples = [gpu_before]
        history = wait_for_completion(prompt_id, gpu_samples)

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
        final_path = expected_variant_output(base_name, variant_index, final_ext)

        shutil.copy2(video_file, final_path)

        elapsed = time.monotonic() - started_at
        gpu_after = get_gpu_snapshot()
        gpu_samples.append(gpu_after)
        gpu_summary = summarize_gpu_samples(gpu_samples)
        gpu_processes_after = get_gpu_process_snapshot()
        append_perf_log({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image": image_path.name,
            "variant": variant_index,
            "seconds": f"{elapsed:.2f}",
            "seed": seed,
            "strength": strength,
            "gpu_before": json.dumps(gpu_before, ensure_ascii=True),
            "gpu_after": json.dumps(gpu_after, ensure_ascii=True),
            **gpu_summary,
            "gpu_processes_before": gpu_processes_before,
            "gpu_processes_after": gpu_processes_after,
            "status": "ok",
        })

        print(f"Saved: {final_path}")
        print(f"Elapsed: {elapsed:.2f}s")

    except Exception as e:
        elapsed = time.monotonic() - started_at
        gpu_after = get_gpu_snapshot()
        gpu_summary = summarize_gpu_samples([gpu_before, gpu_after])
        append_perf_log({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image": image_path.name,
            "variant": variant_index,
            "seconds": f"{elapsed:.2f}",
            "seed": seed,
            "strength": strength,
            "gpu_before": json.dumps(gpu_before, ensure_ascii=True),
            "gpu_after": json.dumps(gpu_after, ensure_ascii=True),
            **gpu_summary,
            "gpu_processes_before": gpu_processes_before,
            "gpu_processes_after": get_gpu_process_snapshot(),
            "status": "error",
            "error": str(e),
        })
        raise

    finally:
        if FREE_COMFY_MEMORY_AFTER_EACH_IMAGE:
            try:
                free_comfy_memory()
                print("ComfyUI memory freed")
            except Exception as e:
                print(f"[WARN] Could not free ComfyUI memory: {e}")

def process_image(image_path, workflow_template, image_index, total_images):

    print(f"\n=== [{image_index}/{total_images}] Processing: {image_path.name} ===")

    base_name = image_path.stem
    pending_variants = []

    for variant_index in range(1, VARIANTS_PER_IMAGE + 1):
        existing = find_existing_variant_output(base_name, variant_index)
        if existing:
            print(f"  variant {variant_index}: already done -> {existing.name}")
            continue
        pending_variants.append(variant_index)

    if not pending_variants:
        print("  all variants already done")
        return

    comfy_name = copy_image_to_comfy_input(image_path)

    for variant_index in pending_variants:
        process_variant(image_path, comfy_name, workflow_template, variant_index)

def main():

    ensure_dirs()

    workflow = load_workflow()
    images = get_input_images()

    print(f"Found {len(images)} images")

    for image_index, image in enumerate(images, start=1):

        try:
            process_image(image, workflow, image_index, len(images))
        except Exception as e:
            print(f"[ERROR] {image.name}: {e}")

    print("\nDone")

if __name__ == "__main__":
    main()
