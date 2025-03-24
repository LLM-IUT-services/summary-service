import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app 

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_summarize_text(client):
    response = client.post('/api/summarize', json={"text": "Flask is a micro web framework."})
    assert response.status_code == 200
    assert "summary" in response.get_json()

def test_summarize_text_missing_field(client):
    response = client.post('/api/summarize', json={})
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_answer_question(client):
    response = client.post('/api/qa', json={"question": "Who built the Eiffel Tower?", "context": "The Eiffel Tower was designed by Gustave Eiffel."})
    assert response.status_code == 200
    assert "answer" in response.get_json()

def test_answer_question_missing_field(client):
    response = client.post('/api/qa', json={"question": "Who built the Eiffel Tower?"})
    assert response.status_code == 400
    assert "error" in response.get_json()
