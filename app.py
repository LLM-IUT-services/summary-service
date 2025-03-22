from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
from flask_restx import Api, Resource, fields
from LLM_base import FaSummarizationPipeline, FaQA_Pipeline, EnPipeline

app = Flask(__name__)
api = Api(app, version='1.0', title='Summary Service API',
          description='A simple API for text summarization, QA, and event reporting',
          doc='/swagger') 

ns = api.namespace('api', description='API operations')

fa_summarization_pipeline = FaSummarizationPipeline()
fa_qa_pipeline = FaQA_Pipeline()
en_pipeline = EnPipeline()

summarize_model = api.model('Summarize', {
    'text': fields.String(required=True, description='Text to summarize'),
    'max_length': fields.Integer(required=False, description='Maximum length of the summary')
})

qa_model = api.model('QA', {
    'question': fields.String(required=True, description='Question to answer'),
    'context': fields.String(required=False, description='Context for the question')
})

@ns.route('/summarize')
class Summarize(Resource):
    @ns.expect(summarize_model)
    def post(self):
        data = request.json
        text = data.get('text')
        max_length = data.get('max_length', 256)
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        summary = en_pipeline.summarize(text, model_max_length=max_length)
        return jsonify({'summary': summary})

@ns.route('/qa')
class QA(Resource):
    @ns.expect(qa_model)
    def post(self):
        data = request.json
        question = data.get('question')
        context = data.get('context', '')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        answer = en_pipeline.QA(context, question)
        return jsonify({'answer': answer})

@ns.route('/event-report')
class EventReport(Resource):
    def post(self):
        return jsonify({'message': 'Event report endpoint'}), 200

if __name__ == '__main__':
    app.run(debug=True)