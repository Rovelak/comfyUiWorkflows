import copy
import json
import mimetypes
import random
import re
import sys
import time
import uuid
from pathlib import Path

import requests

# =========================
# CONFIG
# =========================

COMFYUI_URL = "http://127.0.0.1:8188"

WORKFLOW_PATH = Path(r"D:\IA\comfyUiWorkflows\ltx23\florence-caption.json")

INPUT_IMAGES_DIR = Path(r"D:\Home\images_a_animer")

FINAL_OUTPUT_DIR = Path(r"D:\Home\final_videos")

PROMPT_OUTPUT_DIR = FINAL_OUTPUT_DIR / "ltx23_prompts"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MAX_IMAGES = None

OVERWRITE_EXISTING = False

# Rebuilds prompt JSONs from the existing Florence caption without calling Florence2 again.
# Useful when improving the prompt rules after captions have already been generated.
REFRESH_EXISTING_PROMPTS_FROM_CAPTION = True

FLORENCE_TASK = "more_detailed_caption"

BASE_FIDELITY_PROMPT = (
    "Animate this single still image into one continuous shot. "
    "Preserve the exact same scene, same people, same clothing, same environment, "
    "same composition, and same camera angle."
)

NEGATIVE_PROMPT = (
    "scene change, cut, transition, different shot, different location, different person, "
    "extra person, subject replacement, face change, identity change, body change, pose replacement, "
    "clothing change, gender swap, age change, object deformation, product deformation, logo change, "
    "text change, unreadable text, warping, morphing, melting, hallucinated background, new objects, "
    "new characters, strong motion, camera shake, flicker, jitter, blur, low quality, watermark, "
    "subtitles, overlay, text"
)


# =========================
# COMFYUI HELPERS
# =========================

def wait_for_server(timeout_s=300):
    started_at = time.monotonic()
    while True:
        try:
            response = requests.get(f"{COMFYUI_URL}/history", timeout=5)
            if response.status_code < 500:
                return
        except requests.RequestException:
            pass

        if time.monotonic() - started_at > timeout_s:
            raise TimeoutError(f"ComfyUI ne repond pas sur {COMFYUI_URL}")

        time.sleep(2)


def upload_image(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "application/octet-stream"

    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, mime_type)}
        data = {"type": "input", "overwrite": "true"}
        response = requests.post(f"{COMFYUI_URL}/upload/image", files=files, data=data, timeout=120)
        response.raise_for_status()
        payload = response.json()

    return payload.get("name") or payload.get("filename") or image_path.name


def queue_prompt(prompt_workflow: dict) -> str:
    payload = {
        "prompt": prompt_workflow,
        "client_id": str(uuid.uuid4()),
    }
    response = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout_s=1800) -> dict:
    started_at = time.monotonic()

    while True:
        response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=60)
        response.raise_for_status()
        history = response.json()

        if prompt_id in history:
            return history[prompt_id]

        if time.monotonic() - started_at > timeout_s:
            raise TimeoutError(f"Timeout en attendant Florence2 pour prompt_id={prompt_id}")

        time.sleep(2)


def load_workflow() -> dict:
    with WORKFLOW_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_workflow(workflow_template: dict, image_filename: str, seed: int) -> dict:
    workflow = copy.deepcopy(workflow_template)

    for node in workflow.values():
        if node.get("class_type") == "LoadImage":
            node.setdefault("inputs", {})
            node["inputs"]["image"] = image_filename

        if node.get("class_type") == "Florence2Run":
            node.setdefault("inputs", {})
            node["inputs"]["task"] = FLORENCE_TASK
            node["inputs"]["seed"] = seed

    return workflow


def find_caption_in_history(history_entry: dict) -> str:
    outputs = history_entry.get("outputs", {})

    preferred_keys = (
        "text",
        "texts",
        "string",
        "strings",
        "caption",
        "captions",
        "STRING",
        "ui",
    )

    def flatten_strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from flatten_strings(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from flatten_strings(item)

    for node_output in outputs.values():
        for key in preferred_keys:
            if key in node_output:
                candidates = [s.strip() for s in flatten_strings(node_output[key]) if s.strip()]
                if candidates:
                    return max(candidates, key=len)

    candidates = []
    for node_output in outputs.values():
        candidates.extend(s.strip() for s in flatten_strings(node_output) if s.strip())

    if candidates:
        return max(candidates, key=len)

    raise RuntimeError(f"Aucune caption texte trouvee dans l'historique: {outputs}")


# =========================
# PROMPT BUILDING
# =========================

def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def contains_any(text: str, words) -> bool:
    for word in words:
        if " " in word or "-" in word:
            if word in text:
                return True
        elif re.search(rf"\b{re.escape(word)}\b", text):
            return True

    return False


def detect_image_type(caption: str) -> str:
    text = caption.lower()

    has_person = contains_any(text, (
        "person", "people", "man", "woman", "child", "baby", "boy", "girl",
        "men", "women", "children", "kids", "boys", "girls", "father", "mother",
        "grandfather", "grandmother", "family", "portrait", "crowd", "tourist",
        "swimmer", "diver", "couple",
    ))
    has_adult = contains_any(text, ("man", "woman", "father", "mother", "grandfather", "grandmother", "adult"))
    has_child = contains_any(text, ("baby", "child", "boy", "girl", "children", "kid", "kids", "son", "daughter"))
    has_animal = contains_any(text, (
        "dog", "puppy", "puppies", "cat", "kitten", "horse", "bird", "pet", "animal", "monkey", "elephant",
        "fish", "dolphin", "shark", "turtle", "sea turtle", "underwater animal",
    ))
    optional_person_prefix = r"(a\s+|an\s+|the\s+)?(young\s+|little\s+|small\s+|older\s+|elderly\s+)?"
    has_multiple_people = (
        contains_any(text, ("group", "crowd", "couple", "several people", "many people", "two people", "three people", "four people", "five people"))
        or re.search(r"\b(two|three|four|five|several|many)\s+(\w+\s+){0,2}(men|women|boys|girls|children|kids|people)\b", text)
        or re.search(rf"\b(man|woman|boy|girl|child)\b\s+and\s+{optional_person_prefix}\b(man|woman|boy|girl|child)\b", text)
        or re.search(rf"\b(man|woman|boy|girl|child)\b.{{0,40}},\s+and\s+{optional_person_prefix}\b(man|woman|boy|girl|child)\b", text)
    )

    if contains_any(text, ("underwater", "diving", "scuba", "coral", "reef")):
        return "underwater"
    if contains_any(text, ("crowd", "many people", "large group", "people in the distance", "distant people")):
        return "distant_crowd"
    if has_person and has_animal:
        return "people_with_animal"
    if has_person and (
        contains_any(text, ("baby", "father", "mother", "grandfather", "grandmother", "family", "lap"))
        or (has_adult and has_child)
    ):
        return "family_people"
    if has_multiple_people:
        return "group_people"
    if has_person and contains_any(text, ("close-up", "close up", "portrait", "headshot", "selfie")):
        return "close_portrait"
    if has_person:
        return "single_person"
    if has_animal:
        return "animal_scene"
    if contains_any(text, ("room", "interior", "kitchen", "bedroom", "living room", "office", "furniture")):
        return "interior"
    if contains_any(text, ("street", "city", "urban", "building", "road", "sidewalk", "traffic")):
        return "street_or_city"
    if contains_any(text, ("mountain", "forest", "beach", "sea", "ocean", "lake", "river", "sky", "landscape", "field")):
        return "landscape"
    if contains_any(text, ("car", "vehicle", "motorcycle", "bike", "truck", "airplane", "boat", "train")):
        return "vehicle"
    if contains_any(text, ("product", "package", "packaging", "box", "watch", "shoe", "bag", "device")):
        return "product"

    return "general"


def detect_motion_hints(caption: str, image_type: str) -> list:
    text = caption.lower()
    hints = []

    if image_type == "distant_crowd":
        hints.extend(["tiny distant crowd motion", "subtle background life"])
    elif contains_any(text, ("man", "woman", "person", "people", "father", "mother", "grandfather", "grandmother")):
        hints.extend(["subtle breathing", "tiny head and shoulder movement"])
    if contains_any(text, ("baby", "child", "boy", "girl")):
        hints.extend(["tiny baby or child movement", "small hand movement"])
    if contains_any(text, ("dog", "puppy", "puppies", "cat", "kitten", "horse", "monkey", "elephant", "pet", "animal")):
        hints.extend(["subtle animal breathing", "tiny ear or head movement"])
    if contains_any(text, ("fish", "dolphin", "shark", "turtle", "underwater", "diving", "scuba")):
        hints.extend(["gentle underwater drift", "soft water movement"])
    if image_type != "distant_crowd" and contains_any(text, ("crowd", "many people", "distant people")):
        hints.extend(["tiny distant crowd motion", "subtle background life"])

    people_types = {"close_portrait", "single_person", "family_people", "people_with_animal", "group_people"}
    if image_type in people_types:
        keyword_hints = [
            (("hair", "beard"), "gentle hair movement"),
            (("dress", "shirt", "coat", "jacket", "clothing", "fabric", "scarf"), "slight fabric movement"),
        ]
    elif image_type == "distant_crowd":
        keyword_hints = [
            (("street", "city", "building", "road", "background"), "soft background parallax"),
        ]
    else:
        keyword_hints = [
            (("hair", "beard"), "gentle hair movement"),
            (("dress", "shirt", "coat", "jacket", "clothing", "fabric", "scarf"), "slight fabric movement"),
            (("tree", "leaf", "leaves", "plant", "grass", "flower"), "subtle leaf or plant movement"),
            (("cloud", "sky"), "soft cloud or sky movement"),
            (("water", "sea", "ocean", "lake", "river", "wave"), "gentle water motion"),
            (("smoke", "steam", "mist", "fog"), "soft atmospheric movement"),
            (("metal", "car", "vehicle", "watch", "reflection"), "delicate reflection movement"),
            (("curtain", "window", "shadow", "sunlight", "lamp", "light"), "delicate light and shadow variation"),
            (("street", "city", "building", "road", "background"), "soft background parallax"),
        ]

    for keywords, hint in keyword_hints:
        if contains_any(text, keywords) and hint not in hints:
            hints.append(hint)

    defaults_by_type = {
        "close_portrait": ["subtle breathing", "tiny facial micro-motion", "gentle hair movement", "stable camera"],
        "single_person": ["subtle breathing", "tiny head and shoulder movement", "slight fabric movement", "stable camera"],
        "family_people": ["subtle breathing", "tiny adult and baby micro-movements", "small hand movement", "stable camera"],
        "people_with_animal": ["subtle human breathing", "tiny baby or child movement", "subtle animal breathing", "tiny ear or head movement", "stable camera"],
        "group_people": ["subtle independent micro-motions", "tiny posture shifts", "gentle fabric movement", "stable camera"],
        "distant_crowd": ["tiny distant crowd motion", "subtle background life", "stable camera"],
        "animal_scene": ["subtle animal breathing", "tiny ear or head movement", "stable camera"],
        "underwater": ["gentle underwater drift", "soft water movement", "subtle floating particles", "stable camera"],
        "landscape": ["soft environmental motion", "slow cinematic parallax", "delicate light variation"],
        "street_or_city": ["soft background parallax", "subtle urban atmosphere", "delicate light variation"],
        "product": ["very slow camera push-in", "delicate reflection movement", "controlled light sweep"],
        "food": ["tiny camera push-in", "soft steam or atmosphere when appropriate", "delicate highlight movement"],
        "interior": ["slow camera parallax", "delicate light and shadow variation", "tiny environmental motion"],
        "vehicle": ["slow camera push-in or slight orbit", "delicate reflection movement", "stable vehicle shape"],
        "general": ["subtle natural motion", "soft parallax", "delicate light variation"],
    }

    for hint in defaults_by_type.get(image_type, defaults_by_type["general"]):
        if hint not in hints:
            hints.append(hint)

    return hints[:5]


def detect_static_elements(caption: str, image_type: str) -> list:
    text = caption.lower()
    static_elements = []

    static_keywords = [
        (("table", "desk"), "table"),
        (("glasses", "wine glasses"), "glasses"),
        (("bottle", "wine bottle"), "bottle"),
        (("plate", "dish", "food", "meal"), "plate and food"),
        (("fence", "wall", "background"), "background"),
        (("chair", "bench"), "chair"),
        (("building", "house", "architecture"), "architecture"),
        (("logo", "label", "text"), "text and labels"),
    ]

    for keywords, label in static_keywords:
        if contains_any(text, keywords) and label not in static_elements:
            static_elements.append(label)

    if image_type in {"family_people", "people_with_animal", "close_portrait", "single_person", "group_people", "distant_crowd"}:
        if contains_any(text, ("table", "desk")) and contains_any(text, ("glass", "bottle", "plate", "dish", "food", "meal")):
            if "tabletop objects" not in static_elements:
                static_elements.append("tabletop objects")
        elif contains_any(text, ("glass", "bottle", "plate", "dish", "food", "meal", "bag", "toy", "book", "umbrella")):
            if "loose objects" not in static_elements:
                static_elements.append("loose objects")
        if "background" not in static_elements:
            static_elements.append("background")

    return static_elements[:5]


def build_prompt_variants(image_type: str, motion_hints: list, static_elements: list) -> list:
    hint_text = ", ".join(motion_hints[:4])
    static_text = ", ".join(static_elements[:4]) if static_elements else "background and non-living objects"
    static_sentence = f"Keep {static_text} still and unchanged."

    type_variants = {
        "close_portrait": [
            f"Animate only the person with subtle natural portrait motion: {hint_text}. {static_sentence} Keep the face, identity, pose, and expression stable.",
            f"Add calm lifelike micro-motion only to the person: {hint_text}. {static_sentence} Preserve facial features and avoid any face or body change.",
            f"Add restrained cinematic portrait movement: {hint_text}. {static_sentence} Keep the result fully faithful to the source image.",
            f"Add minimal natural motion with {hint_text}. {static_sentence} No pose replacement, no clothing change, no identity drift.",
        ],
        "single_person": [
            f"Animate only the person with subtle lifelike motion: {hint_text}. {static_sentence} Preserve identity, pose, clothing, and position.",
            f"Add tiny natural body motion to the person: {hint_text}. {static_sentence} Keep the background and objects fixed.",
            f"Add restrained photo animation to the person only: {hint_text}. {static_sentence} No new gestures, no outfit change, no identity drift.",
            f"Add calm human micro-movements: {hint_text}. {static_sentence} Preserve the exact composition and camera angle.",
        ],
        "family_people": [
            f"Animate only the family subjects with subtle lifelike motion: {hint_text}. {static_sentence} Keep identities, age, pose, and relationship exactly the same.",
            f"Add tiny natural motion to the adult and child or baby: {hint_text}. {static_sentence} Preserve faces, hands, clothing, lap or held position, and body contact.",
            f"Add restrained family-photo animation: {hint_text}. {static_sentence} Do not move or reinterpret objects in the scene.",
            f"Add calm micro-movements to the people only: {hint_text}. {static_sentence} No new gestures, no pose replacement, no identity drift.",
        ],
        "people_with_animal": [
            f"Animate only the people and animal with subtle lifelike motion: {hint_text}. {static_sentence} Preserve identities, pose, contact, and positions.",
            f"Add tiny natural motion to the people and animal: {hint_text}. {static_sentence} Keep all objects, scenery, and furniture fixed.",
            f"Add restrained photo motion to living subjects only: {hint_text}. {static_sentence} Do not add, remove, or replace any person or animal.",
            f"Add calm micro-movements to living subjects only: {hint_text}. {static_sentence} No scene change, no object motion, no identity drift.",
        ],
        "group_people": [
            f"Animate only the people with subtle group portrait motion: {hint_text}. {static_sentence} Preserve every person's identity, position, clothing, and pose.",
            f"Add tiny independent human micro-motions: {hint_text}. {static_sentence} Do not add, remove, or replace anyone.",
            f"Add restrained natural movement to people only: {hint_text}. {static_sentence} Keep all faces, bodies, and spacing stable.",
            f"Add calm group motion with {hint_text}. {static_sentence} No new people, no scene change, no subject drift.",
        ],
        "distant_crowd": [
            f"Animate only the distant crowd with tiny ambient motion: {hint_text}. {static_sentence} Keep the camera, scenery, and architecture stable.",
            f"Add subtle background life to the faraway people: {hint_text}. {static_sentence} Do not change the crowd size or layout.",
            f"Add restrained distant human motion: {hint_text}. {static_sentence} No new people, no foreground changes, no scene change.",
            f"Add calm crowd micro-motion only where people are visible: {hint_text}. {static_sentence} Preserve the exact composition.",
        ],
        "animal_scene": [
            f"Animate only the animal with subtle natural motion: {hint_text}. {static_sentence} Preserve the animal's pose, size, and markings.",
            f"Add tiny animal micro-motion: {hint_text}. {static_sentence} Do not change the animal or background.",
            f"Add restrained animal motion with {hint_text}. {static_sentence} Keep the scene composition stable.",
            f"Add calm lifelike motion to the animal only: {hint_text}. {static_sentence} No new objects, no scene change.",
        ],
        "underwater": [
            f"Add subtle underwater motion: {hint_text}. Preserve the exact subjects, reef, water color, and composition.",
            f"Add calm aquatic drift with {hint_text}. Keep animals, divers, and background positions stable.",
            f"Add restrained underwater ambience: {hint_text}. Do not introduce new sea life, bubbles, or scene changes.",
            f"Add gentle water movement only where plausible: {hint_text}. Preserve all subjects and the camera angle.",
        ],
        "landscape": [
            f"Add gentle environmental motion: {hint_text}. Preserve the exact landscape, framing, and time of day.",
            f"Add calm atmospheric movement: {hint_text}. Keep the scene composition stable and faithful.",
            f"Add slow cinematic nature motion with {hint_text}. Do not introduce new objects or weather changes.",
            f"Add minimal natural landscape animation: {hint_text}. Keep all structures and terrain unchanged.",
        ],
        "product": [
            f"Add premium product-shot motion: {hint_text}. Keep the product shape, text, logo, and proportions unchanged.",
            f"Add clean studio movement: {hint_text}. Preserve all branding, edges, materials, and object geometry.",
            f"Add restrained commercial motion with {hint_text}. No deformation, no label changes, no new objects.",
            f"Add subtle product animation: {hint_text}. Keep the product perfectly recognizable and stable.",
        ],
        "vehicle": [
            f"Add cinematic vehicle motion: {hint_text}. Preserve the vehicle design, wheels, logos, and proportions.",
            f"Add controlled automotive movement: {hint_text}. Keep the body shape stable and avoid deformation.",
            f"Add restrained motion with {hint_text}. No vehicle redesign, no new elements, no camera shake.",
            f"Add premium vehicle-shot animation: {hint_text}. Preserve exact scene, paint, reflections, and composition.",
        ],
    }

    variants = type_variants.get(image_type, [
        f"Add only subtle natural motion to plausible living or environmental elements: {hint_text}. {static_sentence} Keep the result fully faithful to the source image.",
        f"Add restrained movement: {hint_text}. {static_sentence} Preserve all subjects, objects, composition, and environment.",
        f"Add calm image-to-video motion with {hint_text}. {static_sentence} No new elements, no scene change, no reinterpretation.",
        f"Add minimal natural animation: {hint_text}. {static_sentence} Keep identity, pose, clothing, objects, and background stable.",
    ])

    return [f"{BASE_FIDELITY_PROMPT} {variant}" for variant in variants]


def build_prompt_payload(image_path: Path, caption: str) -> dict:
    clean_caption = normalize_text(caption)
    image_type = detect_image_type(clean_caption)
    motion_hints = detect_motion_hints(clean_caption, image_type)
    static_elements = detect_static_elements(clean_caption, image_type)
    prompt_variants = build_prompt_variants(image_type, motion_hints, static_elements)

    return {
        "image": image_path.name,
        "source_path": str(image_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "florence_task": FLORENCE_TASK,
        "caption": clean_caption,
        "image_type": image_type,
        "motion_hints": motion_hints,
        "static_elements": static_elements,
        "prompt": prompt_variants[0],
        "prompt_variants": prompt_variants,
        "negative_prompt": NEGATIVE_PROMPT,
    }


# =========================
# FILES
# =========================

def ensure_dirs():
    PROMPT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_input_images():
    images = sorted(
        p for p in INPUT_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]

    return images


def prompt_json_path(image_path: Path) -> Path:
    return PROMPT_OUTPUT_DIR / f"{image_path.stem}.json"


def save_prompt_payload(payload: dict, output_path: Path):
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# =========================
# MAIN
# =========================

def main():
    ensure_dirs()
    workflow_template = load_workflow()
    images = get_input_images()

    print(f"Found {len(images)} images")
    if not images:
        return

    print(f"Waiting for ComfyUI at {COMFYUI_URL} ...")
    wait_for_server()
    print("ComfyUI ready")

    for index, image_path in enumerate(images, start=1):
        output_path = prompt_json_path(image_path)

        print(f"\n=== [{index}/{len(images)}] {image_path.name} ===")

        if output_path.exists() and not OVERWRITE_EXISTING:
            if REFRESH_EXISTING_PROMPTS_FROM_CAPTION:
                try:
                    existing = json.loads(output_path.read_text(encoding="utf-8"))
                    caption = existing.get("caption", "")
                    if not caption.strip():
                        print(f"  already done -> {output_path.name} (no caption to refresh)")
                        continue

                    payload = build_prompt_payload(image_path, caption)
                    payload["created_at"] = existing.get("created_at", payload["created_at"])
                    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    save_prompt_payload(payload, output_path)

                    print(f"  refreshed from existing caption -> {output_path.name}")
                    print(f"  type={payload['image_type']}")
                    print(f"  hints={', '.join(payload['motion_hints'])}")
                    print(f"  static={', '.join(payload['static_elements'])}")
                    continue

                except Exception as e:
                    print(f"  [WARN] refresh failed, keeping existing file: {e}")
                    continue

            print(f"  already done -> {output_path.name}")
            continue

        try:
            uploaded_name = upload_image(image_path)
            seed = random.randint(1, 10_000_000)
            workflow = prepare_workflow(workflow_template, uploaded_name, seed)

            prompt_id = queue_prompt(workflow)
            print(f"  prompt_id={prompt_id}")

            history = wait_for_completion(prompt_id)
            caption = find_caption_in_history(history)
            payload = build_prompt_payload(image_path, caption)
            save_prompt_payload(payload, output_path)

            print(f"  type={payload['image_type']}")
            print(f"  hints={', '.join(payload['motion_hints'])}")
            print(f"  static={', '.join(payload['static_elements'])}")
            print(f"  saved -> {output_path}")

        except Exception as e:
            print(f"  [ERROR] {image_path.name}: {e}")

    print("\nDone")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
