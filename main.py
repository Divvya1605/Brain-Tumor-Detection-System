from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import os
import json
import numpy as np

from s2h_inference import run_s2h_pipeline, render_panel

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Artifacts produced by train_s2h_ablation.py -- copy the whole
# s2h_ablation_results/ directory contents here (or point MODELS_DIR at it).
# Needed: inference_config.json, hybrid.keras, healthy_mean_coeffs.npy,
#         healthy_std_coeffs.npy
# ---------------------------------------------------------------------------
MODELS_DIR = 'models'

with open(os.path.join(MODELS_DIR, 'inference_config.json')) as f:
    CONFIG = json.load(f)

class_labels = CONFIG['class_names']
IMAGE_SIZE = CONFIG['image_size']
N_THETA = CONFIG['n_theta']
N_PHI = CONFIG['n_phi']
MAX_DEGREE = CONFIG['max_degree']
THETA_SECTOR = tuple(CONFIG['theta_sector'])
PHI_SECTOR = tuple(CONFIG['phi_sector'])
NOVELTY_SCALE = CONFIG.get('novelty_scale', 3.0)

model = load_model(os.path.join(MODELS_DIR, 'hybrid.keras'))
healthy_mean_coeffs = np.load(os.path.join(MODELS_DIR, 'healthy_mean_coeffs.npy'))
healthy_std_coeffs = np.load(os.path.join(MODELS_DIR, 'healthy_std_coeffs.npy'))

UPLOAD_FOLDER = './uploads'
RESULTS_FOLDER = './static/results'
for folder in (UPLOAD_FOLDER, RESULTS_FOLDER):
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER


# ---------------------------------------------------------------------------
# S2H-guided joint pipeline: hemispherical mapping -> sector harmonic
# transform -> classification (hybrid CNN+S2H) -> localization (S2H
# residual anomaly map) -> uncertainty (entropy + S2H novelty)
# ---------------------------------------------------------------------------
def analyze_image(image_path, output_name):
    result = run_s2h_pipeline(
        model, image_path, class_labels, healthy_mean_coeffs, healthy_std_coeffs,
        image_size=IMAGE_SIZE, n_theta=N_THETA, n_phi=N_PHI, max_degree=MAX_DEGREE,
        theta_sector=THETA_SECTOR, phi_sector=PHI_SECTOR, novelty_scale=NOVELTY_SCALE,
    )

    panel_path = os.path.join(app.config['RESULTS_FOLDER'], output_name)
    render_panel(result, panel_path)

    if result["predicted_class"] == "notumor":
        headline = "No Tumor"
    else:
        headline = f"Tumor: {result['predicted_class']}"

    return {
        "headline": headline,
        "confidence": f"{result['confidence'] * 100:.2f}%",
        "certainty": f"{result['certainty'] * 100:.2f}%",
        "uncertainty": f"{result['uncertainty'] * 100:.2f}%",
        "confidence_status": result["confidence_status"],
        "panel_path": f"/static/results/{output_name}",
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            output_name = os.path.splitext(file.filename)[0] + '_panel.png'
            analysis = analyze_image(file_location, output_name)

            return render_template(
                'index.html',
                result=analysis["headline"],
                confidence=analysis["confidence"],
                certainty=analysis["certainty"],
                uncertainty=analysis["uncertainty"],
                confidence_status=analysis["confidence_status"],
                panel_path=analysis["panel_path"],
            )

    return render_template('index.html', result=None)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
