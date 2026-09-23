from pathlib import Path
from flask import Flask, jsonify, send_from_directory
import json

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder="static")


@app.get("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.get("/results.json")
def results():
    path = ROOT / "outputs" / "results.json"
    if not path.exists():
        return jsonify({"error": "Run prepare_data.py, train.py, and evaluate.py first"}), 404
    return jsonify(json.loads(path.read_text()))


@app.get("/images/<path:name>")
def images(name):
    return send_from_directory(ROOT / "outputs" / "images", name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8003, debug=False)
