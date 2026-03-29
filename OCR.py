import os
import io
import re
from pathlib import Path
from PIL import Image
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# ================= CONFIG =================
ENDPOINT = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
KEY = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]

IMAGES_DIR = Path(os.getenv("OCR_IMAGES_DIR", "images/"))
MD_DIR = Path(os.getenv("OCR_MD_DIR", "md"))

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MAX_DIM = 3000
JPEG_QUALITY = 90
# =========================================

# ---- CREATE OUTPUT DIR ----
MD_DIR.mkdir(exist_ok=True)

# ---- CLIENT ----
client = DocumentIntelligenceClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(KEY)
)

# ---- NATURAL SORT ----
def natural_key(name: str):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", name)
    ]

# ---- LOAD & RESIZE IMAGE ----
def load_and_resize_image(path: Path):
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if max(w, h) <= MAX_DIM:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    scale = MAX_DIM / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    return buf

# ---- COLLECT SUBDIRECTORIES OR IMAGES ----
subdirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]

if not subdirs:
    # Fallback if there are images directly in IMAGES_DIR
    image_files = [f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
    if not image_files:
        raise RuntimeError(f"No images or subfolders found in {IMAGES_DIR}.")
    subdirs = [IMAGES_DIR]

# ---- OCR ALL IMAGES BY DIRECTORY ----
for sdir in subdirs:
    output_md = MD_DIR / f"{sdir.name}.md" if sdir != IMAGES_DIR else MD_DIR / f"{IMAGES_DIR.name}.md"
    
    image_files = sorted(
        [f for f in sdir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS],
        key=lambda p: natural_key(p.name)
    )
    
    if not image_files:
        continue
        
    print(f"Processing {len(image_files)} pages for {sdir.name} -> {output_md.name}")
    
    with open(output_md, "w", encoding="utf-8") as out_md:
        for page_num, img_path in enumerate(image_files, start=1):
            image_stream = load_and_resize_image(img_path)

            poller = client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=image_stream,
                output_content_format="markdown"
            )

            result = poller.result()

            out_md.write("\n\n---\n\n")
            out_md.write(f"## Page {page_num}\n\n")
            out_md.write(result.content or "_No content detected._")
