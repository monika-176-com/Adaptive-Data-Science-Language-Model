from flask import Flask, render_template, request
import pandas as pd
import os

from task_detector import detect_task
from model_engine import run_pipeline
from llm_helper import generate_llm_summary

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['dataset']

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath)

        task = detect_task(df)

        results = run_pipeline(df, task)

        summary = generate_llm_summary(task, results)

        return render_template(
            'result.html',
            task=task,
            results=results,
            summary=summary
        )

if __name__ == '__main__':
    app.run(debug=True)