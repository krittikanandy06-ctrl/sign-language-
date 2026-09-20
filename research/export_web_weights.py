#!/usr/bin/env python3
"""
Export ASL 28-Class Keras MLP Model Weights to pure JavaScript for 100% Client-Side Web Execution.
Generates docs/asl_model_weights.js with zero external dependencies.
"""

import os
import json
from tensorflow.keras.models import load_model


def export_weights():
    os.makedirs("docs", exist_ok=True)
    model = load_model("app/model/asl_mlp_model.keras")

    w1, b1 = model.layers[0].get_weights()
    w2, b2 = model.layers[2].get_weights()
    w3, b3 = model.layers[4].get_weights()

    weights_dict = {
        "w1": [[round(float(x), 4) for x in row] for row in w1],
        "b1": [round(float(x), 4) for x in b1],
        "w2": [[round(float(x), 4) for x in row] for row in w2],
        "b2": [round(float(x), 4) for x in b2],
        "w3": [[round(float(x), 4) for x in row] for row in w3],
        "b3": [round(float(x), 4) for x in b3],
    }

    with open("app/model/asl_classes.json", "r", encoding="utf-8") as f:
        classes = json.load(f)

    classes_json = json.dumps(classes)
    weights_json = json.dumps(weights_dict, separators=(",", ":"))

    js_code = (
        "// Auto-generated ASL 28-Class MLP Weights (Pure JavaScript Matrix Forward Pass)\n"
        f"var ASL_CLASSES = {classes_json};\n"
        f"var ASL_WEIGHTS = {weights_json};\n"
        "if (typeof globalThis !== 'undefined') {\n"
        "  globalThis.ASL_CLASSES = ASL_CLASSES;\n"
        "  globalThis.ASL_WEIGHTS = ASL_WEIGHTS;\n"
        "}\n"
    )

    out_path = "docs/asl_model_weights.js"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(js_code)

    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"[SUCCESS] Exported standalone weights to {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    export_weights()
