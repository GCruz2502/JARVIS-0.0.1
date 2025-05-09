import logging
import sys
import os
import json # For pretty printing results

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.intent_processor import IntentProcessor
from core.advanced_nlp import AdvancedNLPProcessor
from config.settings import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Test Cases Definition ---
# Each test case is a dictionary.
# 'id': Unique identifier for the test case.
# 'description': What this test case is trying to achieve.
# 'inputs': A list of input variations (e.g., different languages or phrasings).
#   Each input item has:
#     'lang': 'es' or 'en'.
#     'text': The user utterance.
#     'expected_intent_label': The primary intent label (e.g., 'weather_plugin', 'music_plugin', or a zero-shot label).
#     'expected_entities': List of (text, label) tuples. Order might not be guaranteed, so comparison needs to be flexible.
#     'expected_sentiment': 'POSITIVE', 'NEGATIVE', or 'NEUTRAL' (actual values from models).
#     'is_question': True if it's a question for QA.
#     'qa_context': Context for QA (if applicable).
#     'expected_qa_answer_part': A substring expected in the QA answer (if applicable).
#     'expected_empathetic_response': True if negative sentiment should trigger empathetic phrase.

TEST_CASES = [
    {
        "id": "TC001_Weather_ES_Simple",
        "description": "Test basic weather intent in Spanish with a simple location.",
        "inputs": [
            {
                "lang": "es",
                "text": "¿Qué tiempo hace en París?",
                "expected_intent_label": "weather_plugin", # Assuming zero-shot maps to this or direct handling
                "expected_entities": [("París", "LOC"), ("París", "GPE")], # spaCy might give GPE, HF might give LOC
                "expected_sentiment": "NEUTRAL", # Or based on model output
                "is_question": True,
                "qa_context": None,
                "expected_qa_answer_part": None,
                "expected_empathetic_response": False,
            }
        ]
    },
    {
        "id": "TC002_Music_EN_With_Artist_Ruler",
        "description": "Test music intent in English, expecting EntityRuler for WORK_OF_ART.",
        "inputs": [
            {
                "lang": "en",
                "text": "Play the song 'Bohemian Rhapsody' by Queen",
                "expected_intent_label": "music_plugin",
                "expected_entities": [("'Bohemian Rhapsody'", "WORK_OF_ART"), ("Queen", "ORG")], # Ruler for song, model for Queen
                "expected_sentiment": "NEUTRAL",
                "is_question": False,
                "qa_context": None,
                "expected_qa_answer_part": None,
                "expected_empathetic_response": False,
            }
        ]
    },
    {
        "id": "TC003_Sentiment_ES_Negative",
        "description": "Test negative sentiment detection in Spanish and empathetic response.",
        "inputs": [
            {
                "lang": "es",
                "text": "Estoy muy triste hoy.",
                "expected_intent_label": None, # Or a general fallback
                "expected_entities": [],
                "expected_sentiment": "NEGATIVE",
                "is_question": False,
                "qa_context": None,
                "expected_qa_answer_part": None,
                "expected_empathetic_response": True,
            }
        ]
    },
    {
        "id": "TC004_QA_EN_Simple",
        "description": "Test simple Question Answering in English.",
        "inputs": [
            {
                "lang": "en",
                "text": "What is the capital of France?",
                "expected_intent_label": "qa_fallback", # Assuming QA is a fallback intent
                "expected_entities": [("France", "GPE")],
                "expected_sentiment": "NEUTRAL",
                "is_question": True,
                "qa_context": "France is a country in Europe. Its capital is Paris.",
                "expected_qa_answer_part": "Paris",
                "expected_empathetic_response": False,
            }
        ]
    },
    # More test cases will be added here for:
    # - Zero-shot classification for various intents
    # - More complex entity merging scenarios (Ruler vs HF vs spaCy base)
    # - Different languages and phrasings for each category
    # - Edge cases and potential failure points
]

class NLPEvaluationSuite:
    def __init__(self):
        logger.info("Initializing NLP Evaluation Suite...")
        self.config = load_config()
        # Initialize AdvancedNLPProcessor first if IntentProcessor depends on it being passed
        # Or let IntentProcessor initialize its own instance
        self.intent_processor = IntentProcessor(config=self.config)
        logger.info("IntentProcessor initialized.")

    def run_test_case(self, test_case_data):
        logger.info(f"--- Running Test Case: {test_case_data['id']} ---")
        logger.info(f"Description: {test_case_data['description']}")
        
        results = []

        for input_data in test_case_data["inputs"]:
            logger.info(f"Processing lang='{input_data['lang']}', text='{input_data['text']}'")
            
            # Simulate the context that would be in IntentProcessor
            current_context = {
                "current_lang": input_data["lang"],
                "user_raw_input": input_data["text"]
            }
            if input_data.get("qa_context"):
                 current_context["qa_context_override"] = input_data["qa_context"]


            # Process with IntentProcessor
            # The process method should return a dictionary or object with all relevant NLP results
            # For now, we assume it returns a dict like:
            # {
            #   "final_response": "...",
            #   "intent": "...",
            #   "entities": [("text", "label"), ...],
            #   "sentiment": {"label": "...", "score": ...},
            #   "qa_result": {"answer": "...", "score": ...} (if applicable),
            #   "merged_entities": [...]
            #   "empathetic_response_triggered": True/False
            # }
            # This structure needs to be ensured or adapted from IntentProcessor.process()
            
            # For now, let's mock the call and structure.
            # Actual call: processed_output = self.intent_processor.process(input_data['text'], context=current_context)
            
            # Actual call to the updated IntentProcessor.process method
            # Note: The process method signature is process(self, text). Context is handled internally via self.context.
            # If context override is needed for specific tests, process() signature might need adjustment later.
            processed_output = self.intent_processor.process(input_data['text'])
            # --- End Actual Call ---

            actual_intent = processed_output.get("intent_label")
            actual_entities = processed_output.get("merged_entities", []) # Use merged_entities
            actual_sentiment_data = processed_output.get("sentiment") # Get the full sentiment dict
            actual_sentiment_label = actual_sentiment_data.get("label") if actual_sentiment_data else None
            actual_qa_data = processed_output.get("qa_result") # Get the full QA dict
            actual_qa_answer = actual_qa_data.get("answer") if actual_qa_data else None
            actual_empathetic_response = processed_output.get("empathetic_triggered", False) # Use the direct flag


            # --- Comparisons ---
            intent_match = actual_intent == input_data["expected_intent_label"]

            # Refined Entity comparison: Check for exact match of (text, label) sets, ignoring order.
            # Extract just (text, label) from actual entities for comparison
            # Ensure 'text' and 'label' keys exist in each entity dict
            actual_ents_tuples = set(
                (ent.get('text'), ent.get('label'))
                for ent in actual_entities
                if ent.get('text') is not None and ent.get('label') is not None
            )
            expected_ents_tuples = set(tuple(e) for e in input_data["expected_entities"])
            entities_match = actual_ents_tuples == expected_ents_tuples


            sentiment_match = False
            if actual_sentiment_label and input_data["expected_sentiment"]:
                 sentiment_match = actual_sentiment_label.upper() == input_data["expected_sentiment"].upper()
            elif not actual_sentiment_label and not input_data["expected_sentiment"]:
                 sentiment_match = True # Both are None/empty

            qa_answer_match = False
            # Check if QA was expected and an answer was expected/received
            if input_data.get("is_question"):
                expected_answer_part = input_data.get("expected_qa_answer_part")
                if expected_answer_part and actual_qa_answer:
                    qa_answer_match = expected_answer_part.lower() in actual_qa_answer.lower()
                elif not expected_answer_part and not actual_qa_answer: # Expect no answer, got no answer
                    qa_answer_match = True
                # If QA wasn't expected to yield an answer part, but did, it's a mismatch (handled by qa_answer_match starting as False)
            else:
                 # If it wasn't a question, QA match is true if no answer was produced
                 qa_answer_match = not actual_qa_answer

            # Empathetic response match is direct comparison of boolean flags
            empathetic_match = actual_empathetic_response == input_data["expected_empathetic_response"]


            result_summary = {
                "input_text": input_data["text"],
                "lang": input_data["lang"],
                "intent_expected": input_data["expected_intent_label"],
                "intent_actual": actual_intent,
                "intent_match": intent_match,
                "entities_expected": input_data["expected_entities"],
                "entities_actual": [(e.get('text'), e.get('label')) for e in actual_entities], # Store simplified tuples for readability
                "entities_match": entities_match,
                "sentiment_expected": input_data["expected_sentiment"],
                "sentiment_actual": actual_sentiment_label,
                "sentiment_match": sentiment_match,
                "qa_expected_part": input_data.get("expected_qa_answer_part"),
                "qa_actual": actual_qa_answer,
                "qa_match": qa_answer_match,
                "empathetic_expected": input_data["expected_empathetic_response"],
                "empathetic_actual": actual_empathetic_response,
                "empathetic_match": empathetic_match,
                "full_processed_output": processed_output # Keep for debugging details
            }
            results.append(result_summary)
            logger.info(f"Result for input '{input_data['text']}':\n{json.dumps(result_summary, indent=2, ensure_ascii=False)}")


            result_summary = {
                "input_text": input_data["text"],
                "lang": input_data["lang"],
                "intent_expected": input_data["expected_intent_label"],
                "intent_actual": actual_intent,
                "intent_match": intent_match,
                "entities_expected": input_data["expected_entities"],
                "entities_actual": actual_entities,
                "entities_match": entities_match,
                "sentiment_expected": input_data["expected_sentiment"],
                "sentiment_actual": actual_sentiment_label,
                "sentiment_match": sentiment_match,
                "qa_expected_part": input_data["expected_qa_answer_part"],
                "qa_actual": actual_qa_answer,
                "qa_match": qa_answer_match,
                "empathetic_expected": input_data["expected_empathetic_response"],
                "empathetic_actual": actual_empathetic_response,
                "empathetic_match": empathetic_match,
                "full_processed_output": mock_processed_output # For debugging
            }
            results.append(result_summary)
            logger.info(f"Result for input '{input_data['text']}':\n{json.dumps(result_summary, indent=2, ensure_ascii=False)}")
        
        logger.info(f"--- Finished Test Case: {test_case_data['id']} ---")
        return results

    def run_all_tests(self):
        logger.info("======== Starting Full NLP Evaluation Suite ========")
        all_test_results = []
        for test_case_data in TEST_CASES:
            all_test_results.extend(self.run_test_case(test_case_data))
        
        # Basic summary
        # TODO: Add more detailed metrics (precision, recall, F1 for intent & entities)
        num_inputs_tested = sum(len(tc['inputs']) for tc in TEST_CASES)
        successful_intents = sum(1 for r in all_test_results if r['intent_match'])
        successful_entities = sum(1 for r in all_test_results if r['entities_match'])
        successful_sentiments = sum(1 for r in all_test_results if r['sentiment_match'])
        successful_qa = sum(1 for r in all_test_results if r['qa_match'])
        successful_empathetic = sum(1 for r in all_test_results if r['empathetic_match'])

        logger.info("======== NLP Evaluation Suite Summary ========")
        logger.info(f"Total Inputs Processed: {num_inputs_tested}")
        logger.info(f"Intent Matches: {successful_intents}/{num_inputs_tested}")
        logger.info(f"Entity Set Matches: {successful_entities}/{num_inputs_tested}") # Note: definition of "match" is basic
        logger.info(f"Sentiment Matches: {successful_sentiments}/{num_inputs_tested}")
        logger.info(f"QA Matches: {successful_qa}/{num_inputs_tested}")
        logger.info(f"Empathetic Response Matches: {successful_empathetic}/{num_inputs_tested}")
        logger.info("==============================================")
        
        # You could save all_test_results to a JSON file for detailed review
        # with open("nlp_evaluation_results.json", "w", encoding="utf-8") as f:
        #     json.dump(all_test_results, f, indent=2, ensure_ascii=False)
        # logger.info("Detailed results saved to nlp_evaluation_results.json")

if __name__ == "__main__":
    suite = NLPEvaluationSuite()
    suite.run_all_tests()
