from flask import Flask, request, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename
from LLM_base import EnPipeline, FaSummarizationPipeline, FaQA_Pipeline
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)  # Auto-generates OpenAPI docs

en_pipeline = EnPipeline()
fa_summarizer = FaSummarizationPipeline()
fa_qa = FaQA_Pipeline()

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/summarize', methods=['POST'])
def summarize_text():
    """
    Summarize a given text
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              description: The text to summarize
            max_length:
              type: integer
              default: 100
              description: Maximum length of the summary
    responses:
      200:
        description: A summarized version of the text
        schema:
          type: object
          properties:
            summary:
              type: string
      400:
        description: Bad Request
      500:
        description: Internal Server Error
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data['text']
    max_length = data.get('max_length', 100)

    try:
        summary = en_pipeline.summarize(text, model_max_length=max_length)
        return jsonify({"summary": summary}), 200
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500

@app.route('/api/qa', methods=['POST'])
def answer_question():
    """
    Answer a question based on a given context
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            question:
              type: string
              description: The question to answer
            context:
              type: string
              description: The context for the question
    responses:
      200:
        description: The answer to the question
        schema:
          type: object
          properties:
            answer:
              type: string
      400:
        description: Bad Request
      500:
        description: Internal Server Error
    """
    data = request.get_json()
    if not data or 'question' not in data or 'context' not in data:
        return jsonify({"error": "Missing 'question' or 'context' field"}), 400

    question = data['question']
    context = data['context']

    try:
        answer = en_pipeline.QA(context, question)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": f"QA failed: {str(e)}"}), 500

@app.route('/api/event-report', methods=['POST'])
def process_event_report():
    """
    Process an uploaded event report file (CSV or XLSX)
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The event report file (CSV or XLSX)
    responses:
      200:
        description: Summary of the event file
        schema:
          type: object
          properties:
            event_summary:
              type: string
      400:
        description: Bad Request
      500:
        description: Internal Server Error
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only CSV and XLSX are supported."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join("/tmp", filename)
    file.save(file_path)

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        text = " ".join(df.astype(str).values.flatten())
        summary = fa_summarizer.summarize(text)
        return jsonify({"event_summary": summary}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process event file: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
