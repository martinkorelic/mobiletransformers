"""
Script to evaluate Small Language Models (SLMs) against Large Language Models (LLMs) on medical questions using OpenEHR data.
"""

import datetime
import gc
import json
import time

from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models.llms import GeminiModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv

from mobiletransformers.artifacts.validation import MobileTransformerGenerator
from mobiletransformers.config.settings import get_settings
from mobiletransformers.rag.query import ObjectBoxQueryEngine

load_dotenv()


SLM_TO_TEST = "tinyllama"
TEST_DATA_TYPE = "complex"

CHUNK_DATABASE_DIR = "build/ehr_chunk_db"
DOCUMENT_DATABASE_DIR = "build/ehr_document_db"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_API_KEY = get_settings().gemini_api_key


if SLM_TO_TEST == "tinyllama":
    SLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    SLM_MODEL_NAME = "quant_model.onnx"
    SLM_MODEL_DIR = "build/inference-TinyLlama-1.1B-Chat-v1.0-DQ-int4"
elif SLM_TO_TEST == "phi3":
    SLM_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
    SLM_MODEL_NAME = "quant_model.onnx"
    SLM_MODEL_DIR = "build/inference-Phi-3-mini-4k-instruct-int4"

MAX_RESPONSE_LENGTH = 312

# Test questions

if TEST_DATA_TYPE == "simple":
    test_data = [
        {
            "query": "What medications is this patient currently taking and for what conditions?",
            "relevant_docs": ["medications.txt"],
        },
        {
            "query": "What are this patient's known allergies and how severe are they?",
            "relevant_docs": ["allergies.txt"],
        },
        {
            "query": "How has this patient's HbA1c improved over the past 3 months?",
            "relevant_docs": ["hba1c.txt"],
        },
        {
            "query": "What is this patient's current blood pressure and has it improved this week?",
            "relevant_docs": ["vitals.txt"],
        },
        {
            "query": "Does this patient have any allergies to common pain medications?",
            "relevant_docs": ["allergies.txt"],
        },
        {
            "query": "What does this patient's recent lab results show for cholesterol levels?",
            "relevant_docs": ["labs.txt"],
        },
        {
            "query": "What cardiovascular conditions run in this patient's family?",
            "relevant_docs": ["family_history.txt"],
        },
        {
            "query": "What is this patient's most recent HbA1c and how close are they to target?",
            "relevant_docs": ["hba1c.txt"],
        },
        {
            "query": "What active medical conditions does this patient currently have?",
            "relevant_docs": ["conditions.txt"],
        },
        {
            "query": "What foods should this patient avoid based on their allergies?",
            "relevant_docs": ["allergies.txt"],
        },
    ]
elif TEST_DATA_TYPE == "complex":
    test_data = [
        {
            "query": "Given this patient's current medication regimen and allergies, what should be avoided if they need emergency surgery?",
            "relevant_docs": ["allergies.txt"],
        },
        {
            "query": "Based on the HbA1c trend, predict whether this patient will reach their target by the next scheduled test?",
            "relevant_docs": ["hba1c.txt"],
        },
        {
            "query": "What does the blood pressure progression pattern suggest about medication efficacy and lifestyle compliance?",
            "relevant_docs": ["vitals.txt"],
        },
        {
            "query": "Considering the family history, what additional screening tests should this patient prioritize?",
            "relevant_docs": ["family_history.txt"],
        },
        {
            "query": "How do the lipid improvements correlate with the patient's overall cardiovascular risk reduction strategy?",
            "relevant_docs": ["labs.txt"],
        },
        {
            "query": "What medication interaction risks exist with this patient's current drug regimen?",
            "relevant_docs": ["medications.txt"],
        },
        {
            "query": "Based on disease progression timing, which condition likely triggered the cascade of other diagnoses?",
            "relevant_docs": ["conditions.txt"],
        },
        {
            "query": "What lifestyle modifications can be inferred from the HbA1c improvement pattern?",
            "relevant_docs": ["hba1c.txt"],
        },
        {
            "query": "How does this patient's weight loss trajectory correlate with their blood pressure control?",
            "relevant_docs": ["vitals.txt"],
        },
        {
            "query": "What emergency preparedness considerations are needed given this patient's allergy profile?",
            "relevant_docs": ["allergies.txt"],
        },
        {
            "query": "Based on the timing of medication starts, what treatment prioritization strategy was likely used?",
            "relevant_docs": ["medications.txt"],
        },
        {
            "query": "What does the lab trend pattern suggest about treatment adherence and metabolic response?",
            "relevant_docs": ["labs.txt"],
        },
        {
            "query": "How does this patient's genetic predisposition influence their current treatment outcomes?",
            "relevant_docs": ["family_history.txt"],
        },
        {
            "query": "What early warning signs of treatment resistance can be identified from the condition management data?",
            "relevant_docs": ["conditions.txt"],
        },
        {
            "query": "Based on the diabetes progression timeline, what complications should be monitored most closely?",
            "relevant_docs": ["hba1c.txt"],
        },
    ]

# test_data = test_data[:1]


MEDICAL_ASSISTANT_TEMPLATE = """You are a health assistant. Analyze the patient openEHR data and answer the question concisely and briefly in 1-2 sentences.

openEHR data context: {context}

Question: {question}

Answer:"""


def format_medical_prompt(context, question):
    return MEDICAL_ASSISTANT_TEMPLATE.format(context=context, question=question)


# Initialize query database
document_db = ObjectBoxQueryEngine(DOCUMENT_DATABASE_DIR, EMBEDDING_MODEL_ID)
chunk_db = ObjectBoxQueryEngine(CHUNK_DATABASE_DIR, EMBEDDING_MODEL_ID)

llm_test_data = []

# Get all the relevant context based on queries
for t in test_data:
    d_list = document_db.get_by_document(t["relevant_docs"][0])
    c_list = chunk_db.vector_similarity_search(t["query"], top_k=2)

    t["document_context"] = d_list[0].content
    t["chunked_context"] = "\n".join([c.content for c in c_list])
    llm_test_data.append(t)

del document_db
del chunk_db

gc.collect()


# Initialize models
slm_generator = MobileTransformerGenerator(
    model_id=SLM_MODEL_ID, model_name=SLM_MODEL_NAME, model_dir=SLM_MODEL_DIR
)

llm_generator = GeminiModel(model_name="gemini-2.0-flash", api_key=GEMINI_API_KEY)

llm_evaluator = GeminiModel(model_name="gemini-2.5-pro", api_key=GEMINI_API_KEY)

# Initialize metrics
faithfulness_metric = FaithfulnessMetric(threshold=0.9, model=llm_evaluator)

clinical_quality_metric = GEval(
    name="Clinical Quality",
    criteria="Compare the clinical accuracy and completeness of the actual output to the expected output.",
    evaluation_steps=[
        "Check if key medical information from openEHR data is preserved",
        "Compare clinical reasoning and interpretation quality",
        "Assess appropriateness of recommendations or conclusions",
        "Rate overall clinical quality on scale 1-10",
    ],
    threshold=0.7,
    model=llm_evaluator,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Factually incorrect."),
        Rubric(score_range=(3, 6), expected_outcome="Mostly correct."),
        Rubric(score_range=(7, 9), expected_outcome="Correct but missing minor details."),
        Rubric(score_range=(10, 10), expected_outcome="100% correct."),
    ],
)

# Storage for all test cases and responses
all_test_cases = {
    "slm_vs_llm_document": [],
    "slm_vs_llm_chunked": [],
    "document_vs_chunked_slm": [],
    "document_vs_chunked_llm": [],
}

all_responses = []

print("Starting medical LLM evaluation...")
print("=" * 60)

# Generate responses and create test cases
for i, test in enumerate(llm_test_data):
    print(f"Processing test case {i + 1}/{len(llm_test_data)}: {test['query'][:50]}...")

    # Format prompts
    doc_prompt = format_medical_prompt(test["document_context"], test["query"])
    chunk_prompt = format_medical_prompt(test["chunked_context"], test["query"])

    # Generate responses
    slm_response_doc = slm_generator.generate(doc_prompt, max_length=MAX_RESPONSE_LENGTH)
    slm_response_chunk = slm_generator.generate(chunk_prompt, max_length=MAX_RESPONSE_LENGTH)

    llm_response_doc = llm_generator.generate(doc_prompt)[0]
    time.sleep(2)
    llm_response_chunk = llm_generator.generate(chunk_prompt)[0]
    time.sleep(2)

    # Store responses for analysis
    response_data = {
        "test_id": i + 1,
        "query": test["query"],
        "document_context": test["document_context"],
        "chunked_context": test["chunked_context"],
        "responses": {
            "slm_document": slm_response_doc,
            "slm_chunked": slm_response_chunk,
            "llm_document": llm_response_doc,
            "llm_chunked": llm_response_chunk,
        },
    }
    all_responses.append(response_data)

    # 1. SLM vs LLM (Document Context)
    test_case_1 = LLMTestCase(
        input=test["query"],
        actual_output=slm_response_doc,
        expected_output=llm_response_doc,
        retrieval_context=[test["document_context"]],
    )
    all_test_cases["slm_vs_llm_document"].append(test_case_1)

    # 2. SLM vs LLM (Chunked Context)
    test_case_2 = LLMTestCase(
        input=test["query"],
        actual_output=slm_response_chunk,
        expected_output=llm_response_chunk,
        retrieval_context=[test["chunked_context"]],
    )
    all_test_cases["slm_vs_llm_chunked"].append(test_case_2)

    # 3. Document vs Chunked (SLM)
    test_case_3 = LLMTestCase(
        input=test["query"],
        actual_output=slm_response_chunk,
        expected_output=slm_response_doc,
        retrieval_context=[test["chunked_context"]],
    )
    all_test_cases["document_vs_chunked_slm"].append(test_case_3)

    # 4. Document vs Chunked (LLM)
    test_case_4 = LLMTestCase(
        input=test["query"],
        actual_output=llm_response_chunk,
        expected_output=llm_response_doc,
        retrieval_context=[test["chunked_context"]],
    )
    all_test_cases["document_vs_chunked_llm"].append(test_case_4)

print(f"Generated responses for {len(llm_test_data)} test cases")
print("Starting evaluations...")

# Run all 4 evaluations
evaluation_results = {}

for comparison_name, test_cases in all_test_cases.items():
    print(f"\n{'=' * 20} {comparison_name.upper()} {'=' * 20}")
    print(f"Evaluating {len(test_cases)} test cases...")

    # Run evaluation
    results = evaluate(test_cases=test_cases, metrics=[faithfulness_metric, clinical_quality_metric])

    # Extract scores and detailed results - Correct DeepEval API
    faithfulness_scores = []
    clinical_quality_scores = []
    detailed_results = []

    for i, test_result in enumerate(results.test_results):
        test_case_detail = {
            "test_case_id": i + 1,
            "query": test_cases[i].input,
            "actual_output": test_cases[i].actual_output,
            "expected_output": test_cases[i].expected_output,
            "context": test_cases[i].retrieval_context[0] if test_cases[i].retrieval_context else "",
            "metrics": {},
        }

        for metric_data in test_result.metrics_data:
            if "Faithfulness" in metric_data.name:
                faithfulness_scores.append(metric_data.score)
                test_case_detail["metrics"]["faithfulness"] = {
                    "score": metric_data.score,
                    "reason": metric_data.reason,
                    "success": metric_data.success,
                    "threshold": metric_data.threshold,
                }
            elif "Clinical Quality" in metric_data.name:
                clinical_quality_scores.append(metric_data.score)
                test_case_detail["metrics"]["clinical_quality"] = {
                    "score": metric_data.score,
                    "reason": metric_data.reason,
                    "success": metric_data.success,
                    "threshold": metric_data.threshold,
                }

        detailed_results.append(test_case_detail)

    # Calculate summary statistics
    summary_stats = {
        "total_cases": len(test_cases),
        "detailed_results": detailed_results,  # Add detailed results here
        "faithfulness": {
            "average": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
            "pass_rate": sum(1 for s in faithfulness_scores if s >= 0.9) / len(faithfulness_scores)
            if faithfulness_scores
            else 0,
            "scores": faithfulness_scores,
        },
        "clinical_quality": {
            "average": sum(clinical_quality_scores) / len(clinical_quality_scores)
            if clinical_quality_scores
            else 0,
            "pass_rate": sum(1 for s in clinical_quality_scores if s >= 7.0) / len(clinical_quality_scores)
            if clinical_quality_scores
            else 0,
            "scores": clinical_quality_scores,
        },
    }

    evaluation_results[comparison_name] = summary_stats

    # Print results
    print(
        f"Faithfulness - Average: {summary_stats['faithfulness']['average']:.3f}, Pass Rate: {summary_stats['faithfulness']['pass_rate']:.1%}"
    )
    print(
        f"Clinical Quality - Average: {summary_stats['clinical_quality']['average']:.1f}, Pass Rate: {summary_stats['clinical_quality']['pass_rate']:.1%}"
    )

# Print comprehensive results
print("\n" + "=" * 60)
print("COMPREHENSIVE EVALUATION RESULTS")
print("=" * 60)

for comparison_name, stats in evaluation_results.items():
    print(f"\n{comparison_name.replace('_', ' ').title()}:")
    print(
        f"  Faithfulness: {stats['faithfulness']['average']:.3f} (Pass: {stats['faithfulness']['pass_rate']:.1%})"
    )
    print(
        f"  Clinical Quality: {stats['clinical_quality']['average']:.1f} (Pass: {stats['clinical_quality']['pass_rate']:.1%})"
    )

# Prepare data for JSON export
export_data = {
    "evaluation_metadata": {
        "total_test_cases": len(llm_test_data),
        "max_response_length": MAX_RESPONSE_LENGTH,
        "faithfulness_threshold": 0.9,
        "clinical_quality_threshold": 0.7,
    },
    "evaluation_results": evaluation_results,
    "test_responses": all_responses,
    "comparison_descriptions": {
        "slm_vs_llm_document": "Small Language Model vs Large Language Model using full document context",
        "slm_vs_llm_chunked": "Small Language Model vs Large Language Model using chunked context",
        "document_vs_chunked_slm": "Full document vs chunked context for Small Language Model",
        "document_vs_chunked_llm": "Full document vs chunked context for Large Language Model",
    },
}

# Save results to JSON
output_filename = f"medical_llm_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(export_data, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {output_filename}")

# Summary insights
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

# Compare SLM vs LLM performance
slm_llm_doc = evaluation_results["slm_vs_llm_document"]
slm_llm_chunk = evaluation_results["slm_vs_llm_chunked"]

print("1. SLM Performance vs LLM:")
print(
    f"   Document Context - Faithfulness: {slm_llm_doc['faithfulness']['average']:.3f}, Quality: {slm_llm_doc['clinical_quality']['average']:.1f}"
)
print(
    f"   Chunked Context  - Faithfulness: {slm_llm_chunk['faithfulness']['average']:.3f}, Quality: {slm_llm_chunk['clinical_quality']['average']:.1f}"
)

# Compare context types
doc_chunk_slm = evaluation_results["document_vs_chunked_slm"]
doc_chunk_llm = evaluation_results["document_vs_chunked_llm"]

print("\n2. Context Type Impact:")
print(
    f"   SLM: Document vs Chunked - Faithfulness: {doc_chunk_slm['faithfulness']['average']:.3f}, Quality: {doc_chunk_slm['clinical_quality']['average']:.1f}"
)
print(
    f"   LLM: Document vs Chunked - Faithfulness: {doc_chunk_llm['faithfulness']['average']:.3f}, Quality: {doc_chunk_llm['clinical_quality']['average']:.1f}"
)

print(f"\nEvaluation complete! Check {output_filename} for detailed results.")
