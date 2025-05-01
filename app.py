from flask import Flask, request, jsonify
import os
import uuid
import pandas as pd
from werkzeug.utils import secure_filename
from LLM_base import EnPipeline, FaSummarizationPipeline, FaQA_Pipeline
from flasgger import Swagger
from event import process_event_file 
from flask_cors import CORS
from STT import SpeechToText

ALLOWED_AUDIO_EXTENSIONS = {'mp3','wav'}
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app = Flask(__name__)
swagger = Swagger(app)  
CORS(app)

en_pipeline = EnPipeline()
fa_summarizer = FaSummarizationPipeline()
fa_qa = FaQA_Pipeline()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


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
            lang:
              type: string
              enum: [en, fa]
              default: en
              description: Language of the text (English or Persian)
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
    if not data or 'text' not in data or 'lang' not in data:
        return jsonify({"error": "Missing 'text' or 'lang' field"}), 400

    text = data['text']
    lang = data['lang']
    max_length = data.get('max_length', 100)

    try:
        if lang == 'en':
            summary = en_pipeline.summarize(text, model_max_length=max_length)
        elif lang == 'fa':
            summary = fa_summarizer.summarize(text, model_max_length=max_length)
        else:
            return jsonify({"error": "Invalid language. Use 'en' or 'fa'"}), 400

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
            lang:
              type: string
              enum: [en, fa]
              default: en
              description: Language of the question and context (English or Persian)
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
    if not data or 'question' not in data or 'context' not in data or 'lang' not in data:
        return jsonify({"error": "Missing 'question', 'context', or 'lang' field"}), 400

    question = data['question']
    context = data['context']
    lang = data['lang']

    try:
        if lang == 'en':
            answer = en_pipeline.QA(context, question)
        elif lang == 'fa':
            answer = fa_qa.QA(context, question)
        else:
            return jsonify({"error": "Invalid language. Use 'en' or 'fa'"}), 400

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
        result = process_event_file(file_path)
        return jsonify({"event_summary": result}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process event file: {str(e)}"}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe an uploaded MP3 audio file to text
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The MP3 audio file to transcribe
      - name: lang
        in: formData
        type: string
        enum: [en, fa]
        required: true
        description: Language of the audio (English or Persian)
    responses:
      200:
        description: Transcribed text from the audio
        schema:
          type: object
          properties:
            transcription:
              type: string
      400:
        description: Bad Request
      500:
        description: Internal Server Error
    """
    if 'file' not in request.files or 'lang' not in request.form:
        return jsonify({"error": "Missing 'file' or 'lang' field"}), 400

    file = request.files['file']
    lang = request.form['lang']

    if file.filename == '' or not allowed_audio_file(file.filename):
        return jsonify({"error": "Invalid file type. Only MP3 is supported."}), 400

    if lang not in ['en', 'fa']:
        return jsonify({"error": "Invalid language. Use 'en' or 'fa'"}), 400

    unique_id = uuid.uuid4().hex
    filename = f"{unique_id}_{secure_filename(file.filename)}"
    file_path = os.path.join("/tmp", filename)
    file.save(file_path)

    try:
        stt = SpeechToText(lang=lang)
        with open(file_path, "rb") as f:
            if file_path.endswith(".wav"):
              transcription = stt.transcribe_wav_fileobj(f)
            else: 
              transcription = stt.transcribe_mp3_fileobj(f)
        return jsonify({"transcription": transcription}), 200
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)  



if __name__ == '__main__':
    app.run(debug=True)
