import argparse
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from tqdm import tqdm
import torch
import shutil

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="Input image folder")
    parser.add_argument("output_dir", type=Path, help="Output folder")
    parser.add_argument("--model", type=str, default="Salesforce/blip-image-captioning-large", help="Caption model")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto detect by default)")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = pipeline("image-to-text", model=args.model, device=0 if device == "cuda" else -1)

    img_files = sorted(
        [p for p in args.input_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    )
    if not img_files:
        print(f"No PNG/JPG/JPEG images found: {args.input_dir}")
        return

    for img_path in tqdm(img_files, desc="Processing"):
        dst_img = args.output_dir / img_path.name
        dst_txt = args.output_dir / f"{img_path.stem}.txt"

        if dst_txt.exists():
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARN] Skipping unreadable image: {img_path} ({e})")
            continue

        out = pipe(image, max_new_tokens=30)
        caption = ""
        if isinstance(out, list) and out and isinstance(out[0], dict):
            caption = (out[0].get("generated_text") or out[0].get("caption") or "").strip()
        else:
            caption = str(out).strip()

        try:
            dst_txt.write_text(caption + "\n", encoding="utf-8")
        except OSError as e:
            print(f"[WARN] Failed to write TXT: {dst_txt} ({e})")
            continue

        try:
            shutil.copy(img_path, dst_img)
        except OSError as e:
            print(f"[WARN] Failed to copy image: {img_path} -> {dst_img} ({e})")

    print(f"Processing completed, processed {len(img_files)} images (existing labels skipped automatically).")

if __name__ == "__main__":
    main()
