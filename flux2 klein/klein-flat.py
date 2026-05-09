import copy
import json
import mimetypes
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import websocket


# =========================
# Configuration
# =========================
COMFYUI_HOST = "127.0.0.1:8188"
WORKFLOW_JSON = (
    Path(__file__).resolve().parent.parent
    / "flux2 klein"
    / "base-flat-api.json"
)

INPUT_DIR = r"D:\Home\input_images"
OUTPUT_DIR = r"D:\Home\batch_outputs"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VARIANTS_PER_IMAGE = 2
MAX_IMAGES: Optional[int] = None

SERVER_WAIT_TIMEOUT_S = 300
PROMPT_TIMEOUT_S = 3600
CLIENT_ID = str(uuid.uuid4())

# Optional prompt overrides. Keep as None to use the text already stored in base.json.
POSITIVE_PROMPT: Optional[str] = None
NEGATIVE_PROMPT: Optional[str] = None


# =========================
# API helpers
# =========================
def wait_for_server(server: str, timeout_s: int = 300) -> None:
    url = f"http://{server}/history"
    start = time.time()

    while True:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code < 500:
                return
        except requests.RequestException:
            pass

        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"ComfyUI did not respond on http://{server} after {timeout_s} s"
            )
        time.sleep(2)


def upload_image(server: str, image_path: Path, overwrite: bool = True) -> str:
    url = f"http://{server}/upload/image"
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "application/octet-stream"

    with image_path.open("rb") as file:
        files = {"image": (image_path.name, file, mime_type)}
        data = {"type": "input", "overwrite": "true" if overwrite else "false"}
        response = requests.post(url, files=files, data=data, timeout=120)
        response.raise_for_status()
        payload = response.json()

    return payload.get("name") or payload.get("filename") or image_path.name


def queue_prompt(server: str, prompt: Dict[str, Any], client_id: str) -> str:
    url = f"http://{server}/prompt"
    data = {"prompt": prompt, "client_id": client_id}
    response = requests.post(url, json=data, timeout=120)
    response.raise_for_status()
    payload = response.json()

    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Unexpected /prompt response: {payload}")
    return prompt_id


def wait_for_completion(
    server: str,
    client_id: str,
    prompt_id: str,
    timeout_s: int = 3600,
) -> None:
    ws_url = f"ws://{server}/ws?clientId={client_id}"
    ws = websocket.create_connection(ws_url, timeout=timeout_s)
    start = time.time()

    try:
        while True:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out while waiting for prompt {prompt_id}")

            raw = ws.recv()
            if isinstance(raw, bytes):
                continue

            msg = json.loads(raw)
            msg_type = msg.get("type")
            data = msg.get("data", {})

            if msg_type == "executing":
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    return

            if msg_type == "execution_error" and data.get("prompt_id") == prompt_id:
                raise RuntimeError(f"ComfyUI execution error: {data}")

    finally:
        ws.close()


def get_history(server: str, prompt_id: str) -> Dict[str, Any]:
    url = f"http://{server}/history/{prompt_id}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def get_image_bytes(server: str, filename: str, subfolder: str, folder_type: str) -> bytes:
    url = f"http://{server}/view"
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    return response.content


# =========================
# Workflow helpers
# =========================
def load_workflow(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def set_loadimage_nodes(prompt: Dict[str, Any], uploaded_filename: str) -> None:
    found = False
    for node in prompt.values():
        if node.get("class_type") == "LoadImage":
            node.setdefault("inputs", {})
            node["inputs"]["image"] = uploaded_filename
            node["inputs"]["upload"] = "image"
            found = True

    if not found:
        raise RuntimeError("No LoadImage node found in the workflow API JSON.")


def set_save_prefix(prompt: Dict[str, Any], prefix: str) -> None:
    found = False
    for node in prompt.values():
        if node.get("class_type") == "SaveImage":
            node.setdefault("inputs", {})
            node["inputs"]["filename_prefix"] = prefix
            found = True

    if not found:
        raise RuntimeError("No SaveImage node found in the workflow API JSON.")


def set_random_noise_seeds(prompt: Dict[str, Any], seed: int) -> None:
    found = False
    for node in prompt.values():
        if node.get("class_type") == "RandomNoise":
            node.setdefault("inputs", {})
            node["inputs"]["noise_seed"] = seed
            found = True

    if not found:
        print("  [info] no RandomNoise node found for seed replacement.")


def set_text_prompt(prompt: Dict[str, Any], text: str, positive: bool) -> None:
    needle = "Positive Prompt" if positive else "Negative Prompt"
    found = False

    for node in prompt.values():
        if node.get("class_type") != "CLIPTextEncode":
            continue

        title = node.get("_meta", {}).get("title", "")
        if needle.lower() not in title.lower():
            continue

        node.setdefault("inputs", {})
        node["inputs"]["text"] = text
        found = True

    if not found:
        label = "positive" if positive else "negative"
        print(f"  [info] no {label} CLIPTextEncode node found for prompt override.")


def collect_output_images(history_payload: Dict[str, Any], prompt_id: str) -> List[Dict[str, str]]:
    if prompt_id not in history_payload:
        raise RuntimeError(f"prompt_id {prompt_id} missing from /history")

    outputs = history_payload[prompt_id].get("outputs", {})
    images: List[Dict[str, str]] = []

    for node_output in outputs.values():
        for image in node_output.get("images", []):
            if {"filename", "subfolder", "type"} <= set(image.keys()):
                images.append(
                    {
                        "filename": image["filename"],
                        "subfolder": image["subfolder"],
                        "type": image["type"],
                    }
                )

    return images


# =========================
# Files / resume
# =========================
def get_input_files(input_dir: Path) -> List[Path]:
    files = [
        path for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    if MAX_IMAGES is not None:
        files = files[:MAX_IMAGES]
    return files


def expected_variant_output(output_dir: Path, stem: str, variant_index: int) -> Path:
    return output_dir / f"{stem}_klein_v{variant_index:02d}.png"


def make_seed(file_index: int, variant_index: int) -> int:
    return int(time.time() * 1000) + variant_index + file_index * 10000


# =========================
# Main
# =========================
def main() -> None:
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow = load_workflow(WORKFLOW_JSON)
    files = get_input_files(input_dir)

    if not files:
        raise RuntimeError(f"No image found in {input_dir}")

    print(f"Waiting for ComfyUI server on http://{COMFYUI_HOST} ...")
    wait_for_server(COMFYUI_HOST, SERVER_WAIT_TIMEOUT_S)
    print("ComfyUI server detected.\n")

    print(f"{len(files)} image(s) to process")

    for file_index, image_path in enumerate(files, start=1):
        print(f"\n=== [{file_index}/{len(files)}] {image_path.name} ===")

        try:
            uploaded_name = upload_image(COMFYUI_HOST, image_path)
            print(f"  upload ok -> {uploaded_name}")
        except Exception as exc:
            print(f"  [ERROR] upload failed: {exc}")
            continue

        for variant_index in range(1, VARIANTS_PER_IMAGE + 1):
            out_path = expected_variant_output(output_dir, image_path.stem, variant_index)

            if out_path.exists():
                print(f"  variant {variant_index}: already done -> {out_path.name}")
                continue

            seed = make_seed(file_index, variant_index)
            prefix = f"{image_path.stem}_klein_v{variant_index:02d}"

            print(f"  variant {variant_index}: seed={seed}")

            prompt = copy.deepcopy(workflow)

            try:
                set_loadimage_nodes(prompt, uploaded_name)
                set_save_prefix(prompt, prefix)
                set_random_noise_seeds(prompt, seed)

                if POSITIVE_PROMPT is not None:
                    set_text_prompt(prompt, POSITIVE_PROMPT, positive=True)
                if NEGATIVE_PROMPT is not None:
                    set_text_prompt(prompt, NEGATIVE_PROMPT, positive=False)

                prompt_id = queue_prompt(COMFYUI_HOST, prompt, CLIENT_ID)
                print(f"    prompt_id: {prompt_id}")

                wait_for_completion(COMFYUI_HOST, CLIENT_ID, prompt_id, PROMPT_TIMEOUT_S)
                history = get_history(COMFYUI_HOST, prompt_id)
                images = collect_output_images(history, prompt_id)

                if not images:
                    print("    [ERROR] no image found in /history")
                    continue

                image_meta = images[0]
                blob = get_image_bytes(
                    COMFYUI_HOST,
                    image_meta["filename"],
                    image_meta["subfolder"],
                    image_meta["type"],
                )

                out_path.write_bytes(blob)
                print(f"    saved -> {out_path}")

            except Exception as exc:
                print(f"    [ERROR] variant {variant_index} failed: {exc}")
                continue

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
