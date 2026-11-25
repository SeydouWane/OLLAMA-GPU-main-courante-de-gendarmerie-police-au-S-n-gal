"""Extraction locale (Ollama + GPU) du contenu manuscrit des pages de main courante.

Usage :
    python extract.py                 # toutes les images du dossier courant
    python extract.py photo1.jpeg ... # images précises

    OCR_MODEL=qwen3-vl:4b-instruct python extract.py   # autre modèle (sortie JSON structurée)

Résultats : dossier extractions/<nom_image>.json et .md
"""
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OCR_MODEL", "glm-ocr")
IS_GLM = MODEL.startswith("glm-ocr")
NUM_CTX = int(os.environ.get("OCR_CTX", "8192" if IS_GLM else "6144"))  # image (~2000-2600 tokens) + réponse
NUM_GPU = int(os.environ.get("OCR_GPU_LAYERS", "99"))  # force toutes les couches sur le GPU
LONG_SIDE = int(os.environ.get("OCR_SIDE", "1600"))  # agrandit les photos WhatsApp (compressées)
OUT_DIR = Path("extractions")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# --- qwen3-vl : sortie JSON contrainte par un schéma -------------------------------------------
QWEN_PROMPT = """Tu es un expert en transcription de registres manuscrits français (main courante de gendarmerie/police au Sénégal).
Cette page est un tableau à 3 colonnes : "NUMERO DE LA MENTION", "HEURE", "OBJET".
Des annotations manuscrites (souvent en rouge ou en biais) figurent dans la marge / colonne de gauche.

Transcris FIDÈLEMENT chaque mention, de haut en bas :
- numero : le numéro (ex. "0135"), ou "" si la mention continue depuis la page précédente.
- heure : ex. "08H15", ou "".
- texte : le texte complet de la colonne OBJET, mot pour mot, sans résumer ni corriger.
- annotations : texte écrit dans la marge pour cette mention (ex. "Vu", "Levée immobilisation"), ou "".
Mets aussi la date et le numéro de page s'ils sont écrits en haut de la page.
Si un mot est illisible, écris [illisible]. N'invente rien."""

QWEN_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "numero_page": {"type": "string"},
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in ("numero", "heure", "texte", "annotations")},
                "required": ["numero", "heure", "texte", "annotations"],
            },
        },
    },
    "required": ["date", "numero_page", "mentions"],
}

# --- glm-ocr : modèle OCR spécialisé, sortie texte ligne par ligne, découpée ensuite par mention ---
GLM_PROMPT = "Text Recognition:"
HEADER_LINES = {"numero", "de la", "mention", "heure", "objet", "numero de la mention"}
NUMERO_RE = re.compile(r"^0\d{3}$")                      # 0135
HEURE_RE = re.compile(r"^\d{1,2}\s*H\s*\d{2}$", re.I)    # 08H15, 08H 15
DATE_RE = re.compile(r"^\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4}$")  # 20/12/2021
PAGE_RE = re.compile(r"^\d{1,3}$")                       # 65


def free_gpu() -> None:
    """Décharge tous les modèles et attend que la VRAM soit libre."""
    for m in requests.get(f"{OLLAMA}/api/ps", timeout=10).json().get("models", []):
        requests.post(f"{OLLAMA}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)
    for _ in range(30):
        loaded = requests.get(f"{OLLAMA}/api/ps", timeout=10).json().get("models", [])
        try:
            used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=10).stdout.split()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            used = 0
        if not loaded and used < 500:
            return
        time.sleep(1)
    print("  attention : la VRAM n'est pas libre")


def load_on_gpu(retries: int = 3) -> None:
    """Charge le modèle et vérifie que son encodeur d'image tourne sur le GPU, sinon le recharge.

    Quand Ollama n'arrive pas à mesurer la VRAM libre ("unable to refresh free memory, using old
    values" dans server.log), il peut se fier à une ancienne valeur trop basse et placer l'encodeur
    d'image sur le CPU (--no-mmproj-offload) : la lecture d'une page passe alors de ~3 s à 60-230 s.
    Une petite image test suffit à le détecter (mesuré sur RTX 4050 : GPU 180-600 tokens/s,
    CPU 11-34 tokens/s).
    """
    buf = io.BytesIO()
    Image.new("RGB", (448, 448), "white").save(buf, format="JPEG")
    probe = base64.b64encode(buf.getvalue()).decode()
    for attempt in range(1, retries + 1):
        free_gpu()
        _, data = ask(probe, num_predict=1)
        rate = data["prompt_eval_count"] / max(data["prompt_eval_duration"] / 1e9, 1e-6)
        if rate > 80:
            print(f"  encodeur d'image sur GPU ({rate:.0f} tokens/s)")
            return
        print(f"  encodeur d'image sur CPU ({rate:.0f} tokens/s) -> rechargement {attempt}/{retries}")
    print("  attention : l'encodeur d'image reste sur le CPU (extraction lente). Redémarrer Ollama peut aider.")


def split_pages(img: Image.Image) -> list[Image.Image]:
    """Une photo de double page (paysage) est coupée en page gauche / page droite."""
    w, h = img.size
    if w / h > 1.25:
        mid = w // 2
        return [img.crop((0, 0, mid, h)), img.crop((mid, 0, w, h))]
    return [img]


def to_b64(img: Image.Image) -> str:
    scale = LONG_SIDE / max(img.size)
    if scale > 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def ask(image_b64: str, num_predict: int = 2000) -> tuple[str, dict]:
    # Mêmes options à chaque appel : un changement (num_ctx, num_gpu) ferait recharger le modèle.
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": GLM_PROMPT if IS_GLM else QWEN_PROMPT, "images": [image_b64]}],
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_gpu": NUM_GPU, "num_predict": num_predict,
                    "repeat_penalty": 1.05},
    }
    if not IS_GLM:
        payload["format"] = QWEN_SCHEMA
    r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=900)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"], data


def parse_glm(text: str) -> dict:
    """Regroupe les lignes OCR en mentions : un numéro 0xxx ouvre une mention, l'heure suit."""
    page = {"date": "", "numero_page": "", "mentions": []}
    current = {"numero": "", "heure": "", "texte": [], "annotations": ""}
    for raw in text.splitlines():
        line = raw.strip()
        at_top = not page["mentions"] and not current["numero"] and not current["texte"]
        if not line or (at_top and line.lower() in HEADER_LINES):  # en-tête imprimé du tableau
            continue
        compact = line.replace(" ", "")
        if NUMERO_RE.match(compact):
            if current["numero"] or current["texte"]:
                page["mentions"].append(current)
            current = {"numero": compact, "heure": "", "texte": [], "annotations": ""}
        elif HEURE_RE.match(line) and not current["heure"] and not current["texte"]:
            current["heure"] = compact.upper()
        elif DATE_RE.match(line) and not page["date"]:
            page["date"] = compact
        elif PAGE_RE.match(line) and not page["numero_page"] and not page["mentions"] and not current["texte"]:
            page["numero_page"] = line
        else:
            current["texte"].append(line)
    if current["numero"] or current["texte"]:
        page["mentions"].append(current)
    for m in page["mentions"]:
        m["texte"] = "\n".join(m["texte"])
    return page


def extract_page(img: Image.Image) -> tuple[dict, dict]:
    content, data = ask(to_b64(img))
    page = parse_glm(content) if IS_GLM else json.loads(content)
    page["texte_brut"] = content if IS_GLM else ""
    stats = {
        "secondes": round((data["total_duration"] - data.get("load_duration", 0)) / 1e9, 1),
        "lecture_image_s": round(data.get("prompt_eval_duration", 0) / 1e9, 1),
        "tokens_image": data.get("prompt_eval_count"),
    }
    return page, stats


def to_markdown(name: str, pages: list[dict]) -> str:
    lines = [f"# {name}", "", f"Modèle : `{MODEL}`", ""]
    for i, p in enumerate(pages, 1):
        lines += [f"## Page {i}  (date : {p.get('date') or '-'} | n° {p.get('numero_page') or '-'})", ""]
        for m in p.get("mentions", []):
            lines.append(f"### Mention {m['numero'] or '(suite de la page précédente)'} — {m['heure'] or '-'}")
            if m.get("annotations"):
                lines.append(f"*Marge : {m['annotations']}*")
            lines += ["", m["texte"].replace("\n", "  \n"), ""]
    return "\n".join(lines)


def main() -> None:
    args = sys.argv[1:]
    files = [Path(a) for a in args] if args else sorted(
        p for p in Path(".").iterdir() if p.suffix.lower() in IMAGE_EXT
    )
    if not files:
        sys.exit("Aucune image trouvée.")
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Modèle : {MODEL}")
    load_on_gpu()

    for f in files:
        t0 = time.perf_counter()
        img = ImageOps.exif_transpose(Image.open(f))
        pages = []
        for i, half in enumerate(split_pages(img), 1):
            page, s = extract_page(half)
            pages.append(page)
            slow = "  <- lent : encodeur d'image sur CPU ? (voir load_on_gpu)" if s["lecture_image_s"] > 20 else ""
            print(f"  {f.name} page {i} : {s['secondes']}s (lecture image {s['lecture_image_s']}s){slow}")
        (OUT_DIR / f"{f.stem}.json").write_text(
            json.dumps({"modele": MODEL, "pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT_DIR / f"{f.stem}.md").write_text(to_markdown(f.name, pages), encoding="utf-8")
        print(f"OK {f.name} -> {OUT_DIR / f.stem}.md  ({time.perf_counter() - t0:.1f}s, chargement compris)")


if __name__ == "__main__":
    main()
