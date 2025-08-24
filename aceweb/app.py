import os
import json
import uuid
import shutil
import requests
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from gradio_client import Client, handle_file

ACE_URL = os.getenv("ACE_URL", "http://127.0.0.1:7865/")

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "generated"
OUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

_client = None
def get_client():
    global _client
    if _client is None:
        _client = Client(ACE_URL)
    return _client

# Landing page
@app.route("/")
def index():
    return render_template("index.html")

# Generate page
@app.route("/generate.html")
def generate_page():
    return render_template("generate.html")

# API endpoint for generation
@app.route("/generate", methods=["POST"])
def generate():
    try:
        if "payload" not in request.form:
            return jsonify({"error": "Missing payload"}), 400

        data = json.loads(request.form["payload"])

        # Optional ref audio
        ref_audio_file = None
        if "ref_audio_input" in request.files and request.files["ref_audio_input"].filename:
            up = request.files["ref_audio_input"]
            tmp_path = OUT_DIR / f"ref_{uuid.uuid4().hex}_{up.filename}"
            up.save(tmp_path)
            ref_audio_file = handle_file(str(tmp_path))

        client = get_client()

        result = client.predict(
            format=data.get("format", "wav"),
            audio_duration=int(data.get("audio_duration", 60)),
            prompt=data.get("prompt", ""),
            lyrics=data.get("lyrics", ""),
            infer_step=int(data.get("infer_step", 10)),
            guidance_scale=float(data.get("guidance_scale", 8.0)),
            scheduler_type=data.get("scheduler_type", "euler"),
            cfg_type=data.get("cfg_type", "apg"),
            omega_scale=float(data.get("omega_scale", 10.0)),
            manual_seeds=data.get("manual_seeds") or None,
            guidance_interval=float(data.get("guidance_interval", 0.5)),
            guidance_interval_decay=float(data.get("guidance_interval_decay", 0.0)),
            min_guidance_scale=float(data.get("min_guidance_scale", 3.0)),
            use_erg_tag=bool(data.get("use_erg_tag", True)),
            use_erg_lyric=bool(data.get("use_erg_lyric", False)),
            use_erg_diffusion=bool(data.get("use_erg_diffusion", True)),
            oss_steps=data.get("oss_steps") or None,
            guidance_scale_text=float(data.get("guidance_scale_text", 0.0)),
            guidance_scale_lyric=float(data.get("guidance_scale_lyric", 0.0)),
            audio2audio_enable=bool(data.get("audio2audio_enable", False)),
            ref_audio_strength=float(data.get("ref_audio_strength", 0.5)),
            ref_audio_input=ref_audio_file,
            lora_name_or_path=data.get("lora_name_or_path", "none"),
            lora_weight=float(data.get("lora_weight", 1.0)),
            api_name="/__call__"
        )

        # Normalize result
        if isinstance(result, (list, tuple)) and result:
            result = result[0]

        def save_from_url(url, suffix):
            outname = f"music_{uuid.uuid4().hex}{suffix}"
            outpath = OUT_DIR / outname
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(outpath, "wb") as f:
                shutil.copyfileobj(r.raw, f)
            return outname

        def copy_from_path(src, suffix):
            ext = Path(src).suffix or suffix
            outname = f"music_{uuid.uuid4().hex}{ext}"
            shutil.copy(src, OUT_DIR / outname)
            return outname

        fmt_suffix = f".{data.get('format','wav')}"
        filename = None

        if hasattr(result, "path") and result.path:
            p = str(result.path)
            filename = save_from_url(p, fmt_suffix) if p.startswith("http") else copy_from_path(p, fmt_suffix)
        elif isinstance(result, str):
            filename = save_from_url(result, fmt_suffix) if result.startswith("http") else copy_from_path(result, fmt_suffix)

        if not filename:
            return jsonify({"error": f"Unexpected model response: {repr(result)}"}), 500

        return jsonify({"download_url": f"/download/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve generated audio
@app.route("/download/<path:fname>")
def download(fname):
    return send_from_directory(str(OUT_DIR), fname, as_attachment=False)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
