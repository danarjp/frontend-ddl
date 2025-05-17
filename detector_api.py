from detectors import *
from evaluator import *
from flask import Flask, request, jsonify
import tempfile
import os
from flask_cors import CORS
from pathlib import Path


app = Flask(__name__)
CORS(app)

@app.route('/ddl/detect', methods=['POST'])
def predict_ddl():
    kalimat = request.form.get('kalimat')
    file = request.files.get('file')

    if not kalimat or not file:
        return jsonify({'error': 'Missing "kalimat" or "file database json" in request'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        detector = DDLDetector()
        detected_label = detector.detect(kalimat=kalimat, filepath_database_json=tmp_path)
    finally:
        os.remove(tmp_path)

    return jsonify({
        'kalimat': kalimat,
        'detected_label': detected_label
    })

@app.route('/ddl/evaluate', methods=['POST'])
def evaluate():
    filepath_dataset = str(Path(__file__).parent / "detectors/ddl/dataset/dataset.xlsx")
    folderpath_database = str(Path(__file__).parent / "detectors/ddl/dataset/")
    evaluator = SingleLabelEvaluator(filepath_dataset=filepath_dataset, folderpath_database=folderpath_database, detector=DDLDetector())
    
    hasil = evaluator.evaluate()
    
    # Konversi DataFrame ke dictionary agar bisa di-JSON-kan
    hasil['metrics'] = hasil['metrics'].to_dict(orient='index')
    
    return jsonify(hasil)

    

if __name__ == '__main__':
    app.run(debug=True)