"""Compare un modèle Ollama sur les 2 demi-pages de la photo, contre la transcription de référence.

Usage (depuis la racine du projet) :
    python benchmark/compare.py glm-ocr
    python benchmark/compare.py qwen3-vl:4b-instruct

Résultats : benchmark/resultats/<modele>_{gauche,droite}.txt et <modele>_scores.json
Métriques :
    rappel     = part des mots de la référence retrouvés par le modèle
    precision  = part des mots du modèle présents dans la référence (faible = mots inventés)
    reperes    = numéros de mention, heures, dates, téléphone, plaque, noms retrouvés à l'identique
"""
import base64
import io
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if len(sys.argv) > 1:
    os.environ["OCR_MODEL"] = sys.argv[1]  # lu par extract.py à l'import (modèle, contexte)
import extract  # noqa: E402

IMAGE = ROOT / "WhatsApp Image 2026-11-25 at 20.38.11.jpeg"
OUT = Path(__file__).resolve().parent / "resultats"
SIDE = 1600

# Même image, même résolution, même consigne "texte libre" pour tous les modèles.
PROMPTS = {
    "glm-ocr": "Text Recognition:",
    "default": ("Transcris mot pour mot tout le texte manuscrit de cette page de registre (main courante, français). "
                "Pour chaque mention donne: numéro, heure, annotation de marge, puis le texte de la colonne OBJET. "
                "Ne résume pas, ne corrige pas. Mot illisible = [illisible]."),
}


def strip_accents(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def words(text: str) -> list[str]:
    w = re.findall(r"[a-z0-9]+", strip_accents(text).replace("[illisible]", " "))
    # retire l'en-tête imprimé du tableau, s'il a été transcrit
    return " ".join(w).replace("numero de la mention heure objet", " ", 1).split()


def found(repere: str, hyp: str) -> bool:
    if len(repere) <= 2:  # nombre court (n° de page) : doit être un mot entier
        return repere in words(hyp)
    squash = lambda s: re.sub(r"\s+", "", strip_accents(s))  # noqa: E731  "08H 15" == "08H15"
    return squash(repere) in squash(hyp)


def score(ref: dict, hyp: str) -> dict:
    r, h = Counter(words(ref["texte"])), Counter(words(hyp))
    common = sum((r & h).values())
    ok = [x for x in ref["reperes"] if found(x, hyp)]
    return {
        "rappel": round(common / max(1, sum(r.values())), 3),
        "precision": round(common / max(1, sum(h.values())), 3),
        "reperes": f"{len(ok)}/{len(ref['reperes'])}",
        "reperes_manques": [x for x in ref["reperes"] if x not in ok],
    }


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage : python benchmark/compare.py <modele>")
    model, ctx = extract.MODEL, extract.NUM_CTX
    prompt = PROMPTS["glm-ocr" if model.startswith("glm-ocr") else "default"]
    ref = json.loads((Path(__file__).parent / "reference.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", model)

    extract.load_on_gpu()  # mêmes options que ci-dessous, sinon Ollama rechargerait le modèle
    img = Image.open(IMAGE)
    w, h = img.size
    results = {"modele": model, "pages": {}}
    for name, page in {"gauche": img.crop((0, 0, w // 2, h)), "droite": img.crop((w // 2, 0, w, h))}.items():
        s = SIDE / max(page.size)
        page = page.resize((round(page.width * s), round(page.height * s)), Image.LANCZOS)
        buf = io.BytesIO()
        page.convert("RGB").save(buf, "JPEG", quality=95)
        r = requests.post(f"{extract.OLLAMA}/api/chat", json={
            "model": model, "stream": False, "keep_alive": "10m",
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(buf.getvalue()).decode()]}],
            "options": {"temperature": 0, "num_ctx": ctx, "num_gpu": extract.NUM_GPU, "num_predict": 2000,
                        "repeat_penalty": 1.05},
        }, timeout=900).json()
        if "error" in r:
            sys.exit(f"{model}: {r['error']}")
        text = r["message"]["content"]
        (OUT / f"{safe}_{name}.txt").write_text(text, encoding="utf-8")
        results["pages"][name] = {
            "secondes": round((r["total_duration"] - r.get("load_duration", 0)) / 1e9, 1),
            "chargement_s": round(r.get("load_duration", 0) / 1e9, 1),
            "lecture_image_s": round(r["prompt_eval_duration"] / 1e9, 1),
            "tokens_image_prompt": r["prompt_eval_count"],
            "tokens_sortie": r["eval_count"],
            "tokens_par_s": round(r["eval_count"] / (r["eval_duration"] / 1e9), 1),
            **score(ref[name], text),
        }
        print(name, results["pages"][name], flush=True)

    ps = requests.get(f"{extract.OLLAMA}/api/ps", timeout=10).json()["models"]
    m = next(x for x in ps if x["name"].startswith(model))
    results["vram_go"] = round(m["size_vram"] / 1e9, 2)
    results["part_gpu"] = f"{100 * m['size_vram'] / m['size']:.0f} %"
    results["contexte"] = ctx
    print("VRAM", results["vram_go"], "Go |", results["part_gpu"], "GPU")
    (OUT / f"{safe}_scores.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
