from __future__ import annotations

import io
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image, ImageEnhance

app = Flask(__name__)
CORS(app)

# =========================================================
# PRESET XMP EMBUTIDO  (Edição CBP Iluminado)
# =========================================================

DEFAULT_XMP = b"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   crs:Exposure2012="0.00"
   crs:Contrast2012="+15"
   crs:Highlights2012="-35"
   crs:Shadows2012="+30"
   crs:Whites2012="0"
   crs:Blacks2012="0"
   crs:Clarity2012="+20"
   crs:Dehaze="+5"
   crs:Vibrance="+10"
   crs:Saturation="0"
   crs:Texture="+5"
   crs:Sharpness="40"
   crs:SharpenRadius="+1.0"
   crs:SharpenDetail="25"
   crs:ColorNoiseReduction="25"
   crs:SaturationAdjustmentOrange="+10"
  />
 </rdf:RDF>
</x:xmpmeta>"""

# =========================================================
# CONVERSÕES
# =========================================================

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

# =========================================================
# LER XMP
# =========================================================

CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"

def _crs(key: str) -> str:
    return f"{{{CRS}}}{key}".lower()

def read_xmp_bytes(xmp_bytes: bytes) -> dict:
    root = ET.fromstring(xmp_bytes)
    values: dict[str, float] = {}
    for elem in root.iter():
        for attr_name, attr_val in elem.attrib.items():
            if not attr_name.startswith(f"{{{CRS}}}"):
                continue
            try:
                values[attr_name.lower()] = float(attr_val.replace("+", ""))
            except (ValueError, TypeError):
                pass
    return values

# Cache do preset padrão — lido uma vez só
_DEFAULT_XMP_VALUES: dict | None = None

def get_default_xmp() -> dict:
    global _DEFAULT_XMP_VALUES
    if _DEFAULT_XMP_VALUES is None:
        _DEFAULT_XMP_VALUES = read_xmp_bytes(DEFAULT_XMP)
    return _DEFAULT_XMP_VALUES

# =========================================================
# APLICAR XMP
# =========================================================

def apply_xmp(pil_img: Image.Image, xmp: dict) -> Image.Image:
    img = pil_img.convert("RGB")

    exposure = xmp.get(_crs("exposure2012"), 0.0)
    if exposure:
        img = ImageEnhance.Brightness(img).enhance(max(0.1, 1 + exposure * 0.15))

    contrast = xmp.get(_crs("contrast2012"), 0.0)
    if contrast:
        img = ImageEnhance.Contrast(img).enhance(max(0.1, 1 + contrast / 100))

    highlights = xmp.get(_crs("highlights2012"), 0.0)
    if highlights:
        arr = np.array(img, dtype=np.float32)
        arr = np.clip(arr + (arr / 255.0) * highlights * 0.35, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    shadows = xmp.get(_crs("shadows2012"), 0.0)
    if shadows:
        arr = np.array(img, dtype=np.float32)
        arr = np.clip(arr + (1.0 - arr / 255.0) * shadows * 0.35, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    saturation = xmp.get(_crs("saturation"), 0.0)
    if saturation:
        img = ImageEnhance.Color(img).enhance(max(0.0, 1 + saturation / 100))

    vibrance = xmp.get(_crs("vibrance"), 0.0)
    if vibrance:
        bgr = pil_to_bgr(img)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        boost = (1.0 - hsv[:, :, 1] / 255.0) * (vibrance / 100) * 50
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + boost, 0, 255)
        img = bgr_to_pil(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))

    clarity = xmp.get(_crs("clarity2012"), 0.0)
    if clarity:
        bgr = pil_to_bgr(img)
        blur = cv2.GaussianBlur(bgr, (0, 0), 10)
        s = clarity / 100 * 0.4
        img = bgr_to_pil(cv2.addWeighted(bgr, 1 + s, blur, -s, 0))

    return img

def sharpen_image(bgr: np.ndarray, amount: float = 40.0) -> np.ndarray:
    strength = 1.0 + (amount / 150) * 0.5
    blur = cv2.GaussianBlur(bgr, (0, 0), 2)
    return cv2.addWeighted(bgr, strength, blur, -(strength - 1), 0)

# =========================================================
# ENDPOINT  POST /process
# =========================================================

@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify(error="Envie o campo 'image'."), 400

    image_file = request.files["image"]

    # usa preset enviado pelo cliente se existir, senão usa o padrão embutido
    if "preset" in request.files and request.files["preset"].filename:
        xmp = read_xmp_bytes(request.files["preset"].read())
    else:
        xmp = get_default_xmp()

    try:
        pil_img = Image.open(image_file.stream).convert("RGB")
        edited = apply_xmp(pil_img, xmp)

        bgr = sharpen_image(pil_to_bgr(edited), xmp.get(_crs("sharpness"), 40.0))
        edited = bgr_to_pil(bgr)

        buf = io.BytesIO()
        filename = image_file.filename or "resultado.jpg"
        ext = Path(filename).suffix.lower()
        fmt = "PNG" if ext == ".png" else "JPEG"
        save_kw = {"format": fmt}
        if fmt == "JPEG":
            save_kw.update(quality=95, subsampling=0)
        edited.save(buf, **save_kw)
        buf.seek(0)

        return send_file(
            buf,
            mimetype=f"image/{'png' if fmt == 'PNG' else 'jpeg'}",
            as_attachment=True,
            download_name=f"resultado_{filename}",
        )

    except Exception as e:
        return jsonify(error=str(e)), 500

# =========================================================
# SERVIR HTML
# =========================================================

@app.route("/")
def index():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<p>Coloque o xmp-processor.html na mesma pasta.</p>"

if __name__ == "__main__":
    print("Servidor rodando em http://localhost:5000")
    app.run(debug=True, port=5000)
