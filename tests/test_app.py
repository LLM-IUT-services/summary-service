import unittest
import json
from app import app

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_summarize(self):
        response = self.app.post('/api/summarize', 
                                 data=json.dumps({'text': 'This is a test text.'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('summary', json.loads(response.data))

    def test_qa(self):
        response = self.app.post('/api/qa', 
                                 data=json.dumps({'question': 'What is AI?', 'context': 'AI stands for Artificial Intelligence.'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('answer', json.loads(response.data))

    # def test_event_report(self):
      

if __name__ == '__main__':
    unittest.main()