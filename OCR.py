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

IMAGES_DIR = Path(r"D:\Projects\Chatbot\images\DTDH\QDHV\Quy định về học vụ và đào tạo - Phiên bản hợp nhất")
MD_DIR = Path("md")

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MAX_DIM = 3000
JPEG_QUALITY = 90
# =========================================

# ---- DERIVE OUTPUT FILE NAME ----
MD_DIR.mkdir(exist_ok=True)
OUTPUT_MD = MD_DIR / f"{IMAGES_DIR.name}.md"

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

# ---- COLLECT IMAGES (NATURAL ORDER) ----
image_files = sorted(
    [f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTS],
    key=lambda p: natural_key(p.name)
)

if not image_files:
    raise RuntimeError("No images found in folder.")

# ---- OCR ALL IMAGES ----
with open(OUTPUT_MD, "w", encoding="utf-8") as out_md:
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
