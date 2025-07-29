import pytest
from .test_nlp_suite import TEST_CASES, NLPEvaluationSuite, NumpyFloatValuesEncoder, logger
import json

@pytest.fixture(scope="module")
def nlp_suite():
    return NLPEvaluationSuite()

@pytest.mark.parametrize("test_case_data", TEST_CASES, ids=[tc["id"] for tc in TEST_CASES])
def test_nlp_evaluation(nlp_suite, test_case_data):
    logger.info("--- Running Test Case: {} ---".format(test_case_data["id"]))
    logger.info("Description: {}".format(test_case_data["description"]))

    for input_data in test_case_data["inputs"]:
        logger.info("Processing lang=\"{}\", text=\"{}\"".format(input_data["lang"], input_data["text"]))

        if nlp_suite.intent_processor.context_manager:
            nlp_suite.intent_processor.context_manager.clear_all_context()
            if input_data.get("qa_context"):
                nlp_suite.intent_processor.context_manager.set_current_turn_data("qa_context_override", input_data["qa_context"])

        processed_output = nlp_suite.intent_processor.process(input_data["text"], lang_hint=input_data["lang"])

        actual_intent = processed_output.get("intent_label")
        actual_entities = processed_output.get("merged_entities", [])
        actual_sentiment_data = processed_output.get("sentiment")
        actual_sentiment_label = actual_sentiment_data.get("label") if actual_sentiment_data else None
        actual_qa_data = processed_output.get("qa_result")
        actual_qa_answer = actual_qa_data.get("answer") if actual_qa_data else None
        actual_empathetic_response = processed_output.get("empathetic_triggered", False)
        actual_plugin_used = processed_output.get("plugin_used")
        actual_final_response = processed_output.get("final_response")

        # Assertions
        assert actual_intent == input_data["expected_intent_label"], \
            "Intent mismatch for {}: Expected {}, Got {}".format(test_case_data["id"], input_data["expected_intent_label"], actual_intent)

        # Entity comparison (flexible order)
        expected_entities_set = set(tuple(e) for e in input_data["expected_entities"])
        actual_entities_set = set(tuple(e) for e in actual_entities)
        assert expected_entities_set.issubset(actual_entities_set), \
            "Entities mismatch for {}: Expected subset {}, Got {}".format(test_case_data["id"], expected_entities_set, actual_entities_set)

        assert actual_sentiment_label == input_data["expected_sentiment"], \
            "Sentiment mismatch for {}: Expected {}, Got {}".format(test_case_data["id"], input_data["expected_sentiment"], actual_sentiment_label)

        if input_data["is_question"] and input_data["qa_context"]:
            assert input_data["expected_qa_answer_part"] in actual_qa_answer, \
                "QA answer part mismatch for {}: Expected '{}' in '{}'".format(test_case_data["id"], input_data["expected_qa_answer_part"], actual_qa_answer)

        if test_case_data["id"] == "TC012_Ollama_Fallback_ES":
            assert actual_plugin_used == "general_chat_fallback", \
                "Plugin used mismatch for {}: Expected GeneralChatFallback_Ollama, Got {}".format(test_case_data["id"], actual_plugin_used)
            assert "error" not in actual_final_response.get("response", "").lower(), \
                "Final response contains error for {}: {}".format(test_case_data["id"], actual_final_response)

        logger.info("Test Case {} - Input '{}' PASSED.".format(test_case_data["id"], input_data["text"]))

    logger.info("--- Test Case {} COMPLETED ---".format(test_case_data["id"]))
