"""
Comprehensive automated tests for ASL Fingerspelling & Sentence Recognition System.
Tests model integrity, autocomplete triage, sentence debouncing, and API endpoints.
"""

import unittest
import numpy as np
import tensorflow as tf
from app import create_app
from app.utils.autocomplete import AutocompleteEngine
from app.utils.real_time_recognition import get_recognizer


class TestASLSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.recognizer = get_recognizer()

    def test_01_model_loading_and_prediction(self):
        """Verify that the ASL model loads and outputs 28 probability values."""
        self.assertIsNotNone(self.recognizer.model)
        self.assertEqual(len(self.recognizer.classes), 28)
        
        # Test synthetic 63-dimensional landmark vector
        dummy_input = np.random.uniform(0.1, 0.9, (1, 63)).astype(np.float32)
        preds = self.recognizer.model.predict(dummy_input, verbose=0)[0]
        self.assertEqual(len(preds), 28)
        self.assertAlmostEqual(float(np.sum(preds)), 1.0, places=4)

    def test_02_autocomplete_engine(self):
        """Verify prefix suggestion mechanics for sign fingerspelling."""
        engine = AutocompleteEngine()
        
        # Test prefix matching
        suggestions = engine.suggest("HEL")
        self.assertIn("HELLO", suggestions)
        self.assertIn("HELP", suggestions)
        
        # Test case insensitivity
        lower_suggestions = engine.suggest("wat")
        self.assertIn("WATER", lower_suggestions)
        
        # Test empty input
        self.assertEqual(engine.suggest(""), [])

    def test_03_sentence_builder_actions(self):
        """Verify manual sentence manipulation (clear, space, backspace, suggestion)."""
        r = self.recognizer
        r.action_clear()
        self.assertEqual(r.current_word, "")
        self.assertEqual(r.sentence_words, [])
        
        # Simulate typing letters: H -> E -> L -> L -> O
        r.current_word = "HELLO"
        self.assertEqual(r.current_word, "HELLO")
        
        # Space action should commit the word
        r.action_space()
        self.assertEqual(r.current_word, "")
        self.assertEqual(r.sentence_words, ["HELLO"])
        
        # Next word
        r.current_word = "WORLD"
        self.assertEqual(r.get_full_sentence(), "HELLO WORLD")
        
        # Backspace should delete character
        r.action_backspace()
        self.assertEqual(r.current_word, "WORL")
        
        # Suggestion selection
        r.action_select_suggestion("WORLD")
        self.assertEqual(r.sentence_words, ["HELLO", "WORLD"])
        self.assertEqual(r.current_word, "")
        self.assertEqual(r.get_full_sentence(), "HELLO WORLD")

    def test_04_api_endpoints(self):
        """Test Flask web routes and JSON responses."""
        # Main Dashboard
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"ASL Fingerspelling & Sentence Studio", res.data)
        
        # Current predictions endpoint
        res = self.client.get("/current_predictions")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("status", data)
        self.assertIn("current_char", data)
        self.assertIn("char_progress", data)
        self.assertIn("sentence", data)
        
        # Sentence action endpoint
        res = self.client.post("/api/sentence/action", json={"action": "clear"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["state"]["sentence"], "")


if __name__ == "__main__":
    unittest.main()
