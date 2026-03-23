import copy
import json
import mimetypes
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import websocket
from PIL import Image, ImageOps


# =========================
# Configuration
# =========================
COMFYUI_HOST = "127.0.0.1:8188"
WORKFLOW_JSON = "workflow_api.json"

INPUT_DIR = r"D:\Home\input_images"
OUTPUT_DIR = r"D:\Home\batch_outputs"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VARIANTS_PER_IMAGE = 4
MAX_IMAGES: Optional[int] = None

SERVER_WAIT_TIMEOUT_S = 300
CLIENT_ID = str(uuid.uuid4())

# Long côté max visé pour la génération
MAX_SIDE = 1024

# Dimensions minimales de sécurité
MIN_SIDE = 512

# Méthode de resize à injecter dans le workflow si le node l'accepte
UPSCALE_METHOD = "lanczos"

# Crop recommandé : disabled pour garder le ratio
CROP_MODE = "disabled"


# =========================
# API helpers
# =========================
def wait_for_server(server: str, timeout_s: int = 300) -> None:
    url = f"http://{server}/history"
    start = time.time()

    while True:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code < 500:
                return
        except requests.RequestException:
            pass

        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"ComfyUI ne répond pas sur http://{server} après {timeout_s} s"
            )
        time.sleep(2)


def upload_image(server: str, image_path: Path, overwrite: bool = True) -> str:
    url = f"http://{server}/upload/image"
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "application/octet-stream"

    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, mime_type)}
        data = {"type": "input", "overwrite": "true" if overwrite else "false"}
        r = requests.post(url, files=files, data=data, timeout=120)
        r.raise_for_status()
        payload = r.json()

    return payload.get("name") or payload.get("filename") or image_path.name


def queue_prompt(server: str, prompt: Dict[str, Any], client_id: str) -> str:
    url = f"http://{server}/prompt"
    data = {"prompt": prompt, "client_id": client_id}
    r = requests.post(url, json=data, timeout=120)
    r.raise_for_status()
    payload = r.json()

    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Réponse /prompt inattendue: {payload}")
    return prompt_id


def wait_for_completion(server: str, client_id: str, prompt_id: str, timeout_s: int = 3600) -> None:
    ws_url = f"ws://{server}/ws?clientId={client_id}"
    ws = websocket.create_connection(ws_url, timeout=timeout_s)
    start = time.time()

    try:
        while True:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timeout en attendant la fin du prompt {prompt_id}")

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
                raise RuntimeError(f"Erreur d'exécution ComfyUI: {data}")

    finally:
        ws.close()


def get_history(server: str, prompt_id: str) -> Dict[str, Any]:
    url = f"http://{server}/history/{prompt_id}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.json()


def get_image_bytes(server: str, filename: str, subfolder: str, folder_type: str) -> bytes:
    url = f"http://{server}/view"
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.content


# =========================
# Workflow helpers
# =========================
def load_workflow(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_loadimage_nodes(prompt: Dict[str, Any], uploaded_filename: str) -> None:
    found = False
    for _, node in prompt.items():
        if node.get("class_type") == "LoadImage":
            node.setdefault("inputs", {})
            node["inputs"]["image"] = uploaded_filename
            node["inputs"]["upload"] = "image"
            found = True

    if not found:
        raise RuntimeError("Aucun noeud LoadImage trouvé dans le workflow API.")


def set_save_prefix(prompt: Dict[str, Any], prefix: str) -> None:
    found = False
    for _, node in prompt.items():
        if node.get("class_type") == "SaveImage":
            node.setdefault("inputs", {})
            node["inputs"]["filename_prefix"] = prefix
            found = True

    if not found:
        raise RuntimeError("Aucun noeud SaveImage trouvé dans le workflow API.")


def set_all_ksampler_seeds(prompt: Dict[str, Any], seed: int) -> None:
    found = False
    for _, node in prompt.items():
        class_type = node.get("class_type")
        if class_type in {"KSampler", "KSamplerAdvanced"}:
            node.setdefault("inputs", {})
            node["inputs"]["seed"] = seed
            found = True

    if not found:
        print("  [info] aucun KSampler trouvé pour remplacer la seed.")


def set_image_scale_nodes(
    prompt: Dict[str, Any],
    width: int,
    height: int,
    upscale_method: str = "lanczos",
    crop: str = "disabled",
) -> None:
    """
    Remplace les dimensions de tous les nodes ImageScale.
    Compatible avec un workflow API où ImageScale expose :
      inputs.width
      inputs.height
      inputs.upscale_method
      inputs.crop
    """
    found = False
    for _, node in prompt.items():
        if node.get("class_type") == "ImageScale":
            node.setdefault("inputs", {})
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
            if "upscale_method" in node["inputs"] or True:
                node["inputs"]["upscale_method"] = upscale_method
            if "crop" in node["inputs"] or True:
                node["inputs"]["crop"] = crop
            found = True

    if not found:
        print("  [info] aucun node ImageScale trouvé. Dimensions non modifiées.")


def collect_output_images(history_payload: Dict[str, Any], prompt_id: str) -> List[Dict[str, str]]:
    if prompt_id not in history_payload:
        raise RuntimeError(f"prompt_id {prompt_id} absent de /history")

    outputs = history_payload[prompt_id].get("outputs", {})
    images: List[Dict[str, str]] = []

    for _, node_output in outputs.items():
        for img in node_output.get("images", []):
            if {"filename", "subfolder", "type"} <= set(img.keys()):
                images.append(
                    {
                        "filename": img["filename"],
                        "subfolder": img["subfolder"],
                        "type": img["type"],
                    }
                )

    return images


# =========================
# Image helpers
# =========================
def round_to_multiple_of_64(value: int) -> int:
    return max(64, int(round(value / 64.0) * 64))


def clamp(value: int, min_value: int) -> int:
    return max(min_value, value)


def compute_target_size(image_path: Path, max_side: int = 1024, min_side: int = 512) -> Tuple[int, int]:
    """
    Lit la taille réelle de l'image en tenant compte de l'orientation EXIF,
    conserve le ratio, fixe le long côté à max_side, et arrondit aux multiples de 64.
    """
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        src_w, src_h = img.size

    if src_w <= 0 or src_h <= 0:
        raise RuntimeError(f"Dimensions invalides pour {image_path}")

    if src_w >= src_h:
        # paysage ou carré
        scale = max_side / src_w
    else:
        # portrait
        scale = max_side / src_h

    tgt_w = int(src_w * scale)
    tgt_h = int(src_h * scale)

    tgt_w = round_to_multiple_of_64(tgt_w)
    tgt_h = round_to_multiple_of_64(tgt_h)

    tgt_w = clamp(tgt_w, min_side)
    tgt_h = clamp(tgt_h, min_side)

    return tgt_w, tgt_h


# =========================
# Fichiers / reprise
# =========================
def get_input_files(input_dir: Path) -> List[Path]:
    files = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
    if MAX_IMAGES is not None:
        files = files[:MAX_IMAGES]
    return files


def expected_variant_output(output_dir: Path, stem: str, variant_index: int) -> Path:
    return output_dir / f"{stem}_ghibli_v{variant_index:02d}.png"


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
        raise RuntimeError(f"Aucune image trouvée dans {input_dir}")

    print(f"Attente du serveur ComfyUI sur http://{COMFYUI_HOST} ...")
    wait_for_server(COMFYUI_HOST, SERVER_WAIT_TIMEOUT_S)
    print("Serveur ComfyUI détecté.\n")

    print(f"{len(files)} image(s) à traiter")

    for file_index, image_path in enumerate(files, start=1):
        print(f"\n=== [{file_index}/{len(files)}] {image_path.name} ===")

        try:
            target_w, target_h = compute_target_size(image_path, MAX_SIDE, MIN_SIDE)
            print(f"  taille cible: {target_w}x{target_h}")
        except Exception as e:
            print(f"  [ERREUR] lecture dimensions impossible: {e}")
            continue

        try:
            uploaded_name = upload_image(COMFYUI_HOST, image_path)
            print(f"  upload ok -> {uploaded_name}")
        except Exception as e:
            print(f"  [ERREUR] upload impossible: {e}")
            continue

        for variant_index in range(1, VARIANTS_PER_IMAGE + 1):
            out_path = expected_variant_output(output_dir, image_path.stem, variant_index)

            if out_path.exists():
                print(f"  variante {variant_index}: déjà faite -> {out_path.name}")
                continue

            seed = int(time.time() * 1000) + variant_index + file_index * 10000
            prefix = f"{image_path.stem}_ghibli_v{variant_index:02d}"

            print(f"  variante {variant_index}: seed={seed}")

            prompt = copy.deepcopy(workflow)

            try:
                set_loadimage_nodes(prompt, uploaded_name)
                set_save_prefix(prompt, prefix)
                set_all_ksampler_seeds(prompt, seed)
                set_image_scale_nodes(
                    prompt,
                    width=target_w,
                    height=target_h,
                    upscale_method=UPSCALE_METHOD,
                    crop=CROP_MODE,
                )

                prompt_id = queue_prompt(COMFYUI_HOST, prompt, CLIENT_ID)
                print(f"    prompt_id: {prompt_id}")

                wait_for_completion(COMFYUI_HOST, CLIENT_ID, prompt_id)
                history = get_history(COMFYUI_HOST, prompt_id)
                images = collect_output_images(history, prompt_id)

                if not images:
                    print("    [ERREUR] aucune image trouvée dans /history")
                    continue

                img_meta = images[0]
                blob = get_image_bytes(
                    COMFYUI_HOST,
                    img_meta["filename"],
                    img_meta["subfolder"],
                    img_meta["type"],
                )

                out_path.write_bytes(blob)
                print(f"    sauvegardé -> {out_path}")

            except Exception as e:
                print(f"    [ERREUR] variante {variant_index} échouée: {e}")
                continue

    print("\nTerminé.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu.")
        sys.exit(130)