import json
import time
from datetime import datetime
from typing import Any

# Import configuration from config module
from config import (
    AZURE_API_VERSION,
    AZURE_DEPLOYMENT_NAME,
    AZURE_MODEL_NAME,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
)
from deepeval import evaluate
from deepeval.metrics import ArenaGEval, GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ArenaTestCase, LLMTestCase, LLMTestCaseParams
from langchain_openai import AzureChatOpenAI


class AzureOpenAIModel(DeepEvalBaseLLM):
    """
    Custom Azure OpenAI model implementation for DeepEval using LangChain.
    This follows the DeepEval documentation pattern for custom LLM integration.
    """

    def __init__(
        self, azure_endpoint: str, api_key: str, deployment_name: str, model_name: str, api_version: str
    ):
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.model_name = model_name
        self.api_version = api_version

        # Initialize the LangChain Azure OpenAI model
        self.model = AzureChatOpenAI(
            openai_api_version=api_version,
            azure_deployment=deployment_name,
            azure_endpoint=azure_endpoint,
            openai_api_key=api_key,
            temperature=0,  # Set to 0 for consistent evaluation
            max_retries=3,  # Enable retries for rate limit errors
            request_timeout=60,  # Increase timeout
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return f"Azure OpenAI {self.model_name}"


class RecommendationEvaluator:
    """
    A class to evaluate and compare recommendation models using DeepEval with Azure OpenAI.
    Supports comparing base and finetuned models with custom recommendation metrics.

    The class uses Azure OpenAI for LLM-as-a-judge evaluation with custom G-Eval metrics
    specifically designed for recommendation tasks. Metrics use 0-10 scoring internally
    but normalize to 0-1 scale for final results.
    """

    def __init__(
        self,
        azure_endpoint: str = AZURE_OPENAI_ENDPOINT,
        api_key: str = AZURE_OPENAI_API_KEY,
        deployment_name: str = AZURE_DEPLOYMENT_NAME,
        model_name: str = AZURE_MODEL_NAME,
        api_version: str = AZURE_API_VERSION,
    ):
        """
        Initialize the evaluator with Azure OpenAI configuration from config module.

        Args:
            azure_endpoint: Azure OpenAI endpoint URL (from config)
            api_key: Azure OpenAI API key (from config)
            deployment_name: Azure deployment name (from config)
            model_name: Model name (from config)
            api_version: API version (from config)
        """
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.model_name = model_name
        self.api_version = api_version

        # Initialize Azure OpenAI model for evaluation
        self.evaluation_model = AzureOpenAIModel(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
            model_name=model_name,
            api_version=api_version,
        )

        # Create evaluation metrics
        self.metrics = self._create_metrics()

        print("RecommendationEvaluator initialized with Azure OpenAI:")
        print(f"  Endpoint: {azure_endpoint}")
        print(f"  Model: {model_name}")
        print(f"  Deployment: {deployment_name}")

    def _create_metrics(self) -> list[GEval]:
        """
        Create custom G-Eval metrics for recommendation evaluation.
        Uses 0-10 scoring scale which gets normalized to 0-1 later.

        Returns:
            List of configured G-Eval metrics
        """

        recommendation_accuracy = GEval(
            name="Recommendation_Accuracy",
            criteria="""
            Evaluate how accurately the predicted answer matches the correct answer for personalized expected output recommendations.
            Use a scale of 0-10 where:
            - 0: Complete mismatch, wrong recommendations, or model just repeated the input question
            - 1-3: Mostly incorrect with some relevant elements
            - 4-6: Partially correct but missing key recommendations or has significant errors
            - 7-8: Mostly accurate with minor differences in wording or presentation
            - 9-10: Highly accurate, covers all key points even if worded differently
            
            CRITICAL PENALTIES:
            - If the predicted answer is just repeating the input question: Score = 0
            - If predicted answer adds completely unrelated content: Heavy penalty (reduce by 3-4 points)
            - If predicted answer misses major recommendation elements: Significant penalty (reduce by 2-3 points)
            - Different wording but same meaning should NOT be penalized
            - Minor formatting differences should NOT be penalized
            """,
            evaluation_steps=[
                "Check if the predicted answer is just repeating the input question - if yes, score = 0",
                "Compare the core recommendations between predicted and correct answers",
                "Assess if major recommendation elements are missing from predicted answer",
                "Evaluate if the predicted answer covers the same key points as correct answer",
                "Consider that different wording with same meaning should not be penalized",
                "Apply penalties for unrelated content or missing major elements",
                "Assign final score 0-10 based on accuracy, completeness, and relevance",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,  # Will be converted from 7/10 to 0.7/1.0
            model=self.evaluation_model,
            async_mode=False,
        )

        recommendation_completeness = GEval(
            name="Recommendation_Completeness",
            criteria="""
            Evaluate how complete the predicted personalized recommendations are compared to the correct personalized output.
            Use a scale of 0-10 where:
            - 0: No relevant recommendations or just repeated input
            - 1-3: Very incomplete, missing most key recommendations
            - 4-6: Somewhat complete but missing several important recommendations
            - 7-8: Mostly complete with minor omissions
            - 9-10: Comprehensive and complete recommendations
            
            Focus on:
            - Coverage of all important recommendation categories mentioned in correct personalized output
            - Inclusion of key details and specifics
            - Completeness of the recommendation set
            - Avoiding significant omissions that would hurt user experience
            """,
            evaluation_steps=[
                "Identify all recommendation categories/types in the correct answer",
                "Check which categories are covered in the predicted answer",
                "Assess the level of detail provided for each recommendation",
                "Identify any missing key recommendations or important details",
                "Calculate coverage percentage of important elements",
                "Consider if omissions would significantly impact user value",
                "Score based on coverage completeness and thoroughness",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.6,  # Will be converted from 6/10 to 0.6/1.0
            model=self.evaluation_model,
            async_mode=False,
        )

        recommendation_relevance = GEval(
            name="Recommendation_Relevance",
            criteria="""
            Evaluate how relevant and appropriate the personalized recommendations are compared to the expected personalized output.
            Use a scale of 0-10 where:
            - 0: Completely irrelevant or just repeated input question
            - 1-3: Mostly irrelevant recommendations that don't address user needs
            - 4-6: Somewhat relevant but not well-targeted to the specific context
            - 7-8: Highly relevant and well-targeted recommendations
            - 9-10: Perfect relevance and contextual appropriateness
            
            Consider:
            - Alignment with the specific user question/context
            - Appropriateness of recommendation types for the domain
            - Contextual understanding demonstrated by the model
            - Usefulness and actionability of recommendations
            """,
            evaluation_steps=[
                "Analyze the input question to understand user needs and context",
                "Evaluate if predicted recommendations directly address the user's question",
                "Check if recommendations are contextually appropriate for the domain",
                "Assess if the model understood the specific recommendation scenario",
                "Verify recommendations are actionable and useful for the user",
                "Consider if recommendations show good understanding of user intent",
                "Score based on relevance, appropriateness, and contextual fit",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,  # Will be converted from 7/10 to 0.7/1.0
            model=self.evaluation_model,
            async_mode=False,
        )

        return [recommendation_accuracy, recommendation_completeness, recommendation_relevance]

    def load_data(self, base_json_path: str, finetuned_json_path: str) -> tuple[list[dict], list[dict]]:
        """
        Load data from both JSON files and ensure they match.

        Expected JSON structure:
        [
            {
                "input_question": "...",
                "predicted_answer": "...",
                "correct_answer": "...",
                "sample_id": 0
            },
            ...
        ]

        Args:
            base_json_path: Path to base model responses JSON
            finetuned_json_path: Path to finetuned model responses JSON

        Returns:
            Tuple of (base_data, finetuned_data) lists with matching sample_ids
        """
        print("Loading data from:")
        print(f"  Base model: {base_json_path}")
        print(f"  Finetuned model: {finetuned_json_path}")

        with open(base_json_path, encoding="utf-8") as f:
            base_data = json.load(f)

        with open(finetuned_json_path, encoding="utf-8") as f:
            finetuned_data = json.load(f)

        print(f"Loaded {len(base_data)} base samples and {len(finetuned_data)} finetuned samples")

        # Ensure both datasets have the same sample_ids
        base_ids = {item["sample_id"] for item in base_data}
        finetuned_ids = {item["sample_id"] for item in finetuned_data}

        if base_ids != finetuned_ids:
            print("Warning: Sample ID mismatch detected!")
            print(f"  Base model samples: {len(base_ids)}")
            print(f"  Finetuned model samples: {len(finetuned_ids)}")

            common_ids = base_ids.intersection(finetuned_ids)
            missing_in_base = finetuned_ids - base_ids
            missing_in_finetuned = base_ids - finetuned_ids

            if missing_in_base:
                print(f"  Missing in base: {sorted(list(missing_in_base))}")
            if missing_in_finetuned:
                print(f"  Missing in finetuned: {sorted(list(missing_in_finetuned))}")

            print(f"  Using {len(common_ids)} common samples for evaluation")

            base_data = [item for item in base_data if item["sample_id"] in common_ids]
            finetuned_data = [item for item in finetuned_data if item["sample_id"] in common_ids]

        # Sort by sample_id to ensure matching order
        base_data.sort(key=lambda x: x["sample_id"])
        finetuned_data.sort(key=lambda x: x["sample_id"])

        print(f"Final dataset: {len(base_data)} matched samples")
        return base_data, finetuned_data

    def _normalize_scores(self, evaluation_results) -> list[dict]:
        """
        Normalize scores from 0-10 scale to 0-1 scale and extract results.

        Args:
            evaluation_results: DeepEval evaluation results

        Returns:
            List of normalized result dictionaries
        """
        normalized_results = []

        for result in evaluation_results.test_results:
            normalized_metrics = {}

            # Use metrics_data instead of metrics_metadata
            for metric_data in result.metrics_data:
                metric_name = metric_data.name

                # Normalize score from 0-10 to 0-1, but handle cases where score might already be 0-1
                if metric_data.score <= 1.0:
                    # Score is already normalized (0-1)
                    normalized_score = metric_data.score
                else:
                    # Score is on 0-10 scale, normalize to 0-1
                    normalized_score = metric_data.score / 10.0

                # Determine threshold for this metric
                metric_threshold = 0.7  # default
                for metric in self.metrics:
                    if metric.name == metric_name:
                        metric_threshold = metric.threshold
                        break

                normalized_metrics[metric_name] = {
                    "score": round(normalized_score, 3),
                    "raw_score": metric_data.score,
                    "reason": metric_data.reason,
                    "success": normalized_score >= metric_threshold,
                }

            normalized_results.append(
                {
                    "input": result.input,
                    "actual_output": result.actual_output,
                    "expected_output": result.expected_output,
                    "metrics": normalized_metrics,
                }
            )

        return normalized_results

    def evaluate_model(self, data: list[dict], model_name: str) -> list[dict]:
        """
        Evaluate a single model's responses using all metrics.

        Args:
            data: List of evaluation samples with required fields
            model_name: Name of the model being evaluated (for logging)

        Returns:
            List of normalized evaluation results
        """
        print(f"\nEvaluating {model_name} model...")
        print(f"  Samples: {len(data)}")
        print(f"  Metrics: {len(self.metrics)}")

        # Create test cases
        test_cases = []
        for item in data:
            test_case = LLMTestCase(
                input=item["input_question"],
                actual_output=item["predicted_answer"],
                expected_output=item["correct_answer"],
            )
            test_cases.append(test_case)

        print("  Running evaluation with Azure OpenAI...")

        # Run evaluation
        evaluation_results = evaluate(
            test_cases=test_cases,
            metrics=self.metrics,
            print_results=False,  # We'll handle our own reporting
            run_async=False,
            max_concurrent=2,
        )

        # Normalize scores from 0-10 to 0-1
        normalized_results = self._normalize_scores(evaluation_results)

        # Add sample_id back to results for tracking
        for i, result in enumerate(normalized_results):
            result["sample_id"] = data[i]["sample_id"]

        print(f"  ✓ Evaluation complete for {model_name} model")
        return normalized_results

    def compare_models(self, base_json_path: str, finetuned_json_path: str) -> dict[str, Any]:
        """
        Compare base and finetuned models and return comprehensive results.

        Args:
            base_json_path: Path to base model responses JSON
            finetuned_json_path: Path to finetuned model responses JSON

        Returns:
            Dictionary containing all evaluation results and detailed comparisons
        """
        print("=" * 80)
        print("STARTING MODEL COMPARISON EVALUATION")
        print("=" * 80)

        # Load and validate data
        base_data, finetuned_data = self.load_data(base_json_path, finetuned_json_path)

        if len(base_data) == 0:
            raise ValueError("No matching samples found between base and finetuned datasets")

        # Evaluate both models
        base_results = self.evaluate_model(base_data, "Base")
        finetuned_results = self.evaluate_model(finetuned_data, "Finetuned")

        # Calculate detailed comparison statistics
        print("\nCalculating comparison statistics...")
        comparison_stats = self._calculate_comparison_stats(base_results, finetuned_results)

        # Prepare comprehensive final results
        final_results = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(base_data),
                "metrics_used": [metric.name for metric in self.metrics],
                "evaluation_model": self.evaluation_model.get_model_name(),
                "azure_endpoint": self.azure_endpoint,
                "deployment_name": self.deployment_name,
                "thresholds": {metric.name: metric.threshold for metric in self.metrics},
            },
            "base_model_results": base_results,
            "finetuned_model_results": finetuned_results,
            "comparison_statistics": comparison_stats,
        }

        print("✓ Evaluation complete!")
        return final_results

    def _calculate_comparison_stats(
        self, base_results: list[dict], finetuned_results: list[dict]
    ) -> dict[str, Any]:
        """
        Calculate simple comparison statistics between models - focusing on accuracy only.

        Args:
            base_results: Evaluation results for base model
            finetuned_results: Evaluation results for finetuned model

        Returns:
            Dictionary with simplified comparison statistics
        """
        stats = {}

        # Get available metric names from the actual results
        if not base_results or not finetuned_results:
            return {"error": "No results to compare"}

        available_metrics = list(base_results[0]["metrics"].keys())
        print(f"Available metrics: {available_metrics}")

        # Calculate stats for each available metric
        for metric_name in available_metrics:
            try:
                # Extract scores for this metric
                base_scores = [result["metrics"][metric_name]["score"] for result in base_results]
                finetuned_scores = [result["metrics"][metric_name]["score"] for result in finetuned_results]

                # Simple averages
                base_avg = sum(base_scores) / len(base_scores)
                finetuned_avg = sum(finetuned_scores) / len(finetuned_scores)
                improvement = finetuned_avg - base_avg

                # Count successes (pass rates)
                base_passed = sum(1 for result in base_results if result["metrics"][metric_name]["success"])
                finetuned_passed = sum(
                    1 for result in finetuned_results if result["metrics"][metric_name]["success"]
                )

                # Clean up metric name for display (remove "(GEval)" suffix)
                display_name = metric_name.replace(" (GEval)", "").replace("_", " ")

                stats[display_name] = {
                    "base_average": round(base_avg, 3),
                    "finetuned_average": round(finetuned_avg, 3),
                    "improvement": round(improvement, 3),
                    "base_passed": f"{base_passed}/{len(base_results)}",
                    "finetuned_passed": f"{finetuned_passed}/{len(finetuned_results)}",
                    "pass_rate_improvement": finetuned_passed - base_passed,
                }

            except KeyError as e:
                print(f"Warning: Could not process metric {metric_name}: {e}")
                continue

        # Overall summary - simple average across all metrics
        metric_keys = [k for k in stats.keys() if k != "overall_summary"]
        if metric_keys:
            overall_base = sum(stats[m]["base_average"] for m in metric_keys) / len(metric_keys)
            overall_finetuned = sum(stats[m]["finetuned_average"] for m in metric_keys) / len(metric_keys)

            stats["overall_summary"] = {
                "base_model_average": round(overall_base, 3),
                "finetuned_model_average": round(overall_finetuned, 3),
                "overall_improvement": round(overall_finetuned - overall_base, 3),
                "winner": "Finetuned"
                if overall_finetuned > overall_base
                else "Base"
                if overall_base > overall_finetuned
                else "Tie",
            }

        return stats

    def arena_comparison(self, base_json_path: str, finetuned_json_path: str) -> dict[str, Any]:
        """
        Run head-to-head arena comparison between base and finetuned models.

        Args:
            base_json_path: Path to base model responses JSON
            finetuned_json_path: Path to finetuned model responses JSON

        Returns:
            Dictionary containing arena comparison results
        """
        print("\n" + "=" * 60)
        print("RUNNING ARENA COMPARISON")
        print("=" * 60)

        # Load and validate data
        base_data, finetuned_data = self.load_data(base_json_path, finetuned_json_path)

        if len(base_data) == 0:
            return {"error": "No matching samples found for arena comparison"}

        arena_metric = ArenaGEval(
            name="Closest_to_Ground_Truth",
            criteria="""
            Choose which output is aligns better to the expected personalized output in terms of:
            - Completeness of recommendations compared to ground truth
            - Accuracy of information matching the expected output
            - Overall alignment with the correct answer
            
            The winner should be the output that best matches what the expected output contains.
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=self.evaluation_model,
        )

        arena_results = []
        finetuned_wins = 0
        base_wins = 0

        # Process each matched sample
        for i, (base_item, finetuned_item) in enumerate(zip(base_data, finetuned_data)):
            # Verify sample IDs match
            if base_item["sample_id"] != finetuned_item["sample_id"]:
                print(
                    f"Warning: Sample ID mismatch at index {i}: {base_item['sample_id']} vs {finetuned_item['sample_id']}"
                )
                continue

            sample_id = base_item["sample_id"]
            print(f"  Arena comparison for sample {i + 1}/{len(base_data)} (ID: {sample_id})...")

            # Create arena test case
            arena_test_case = ArenaTestCase(
                contestants={
                    "Base_Model": LLMTestCase(
                        name="base_model",
                        input=finetuned_item["input_question"],
                        actual_output=base_item["predicted_answer"],
                        expected_output=base_item["correct_answer"],
                    ),
                    "Finetuned_Model": LLMTestCase(
                        name="finetuned_model",
                        input=finetuned_item["input_question"],
                        actual_output=finetuned_item["predicted_answer"],
                        expected_output=finetuned_item["correct_answer"],
                    ),
                }
            )

            try:
                # Run arena comparison
                arena_metric.measure(arena_test_case)

                winner = arena_metric.winner
                reason = arena_metric.reason

                # Count wins
                if winner == "Finetuned_Model":
                    finetuned_wins += 1
                elif winner == "Base_Model":
                    base_wins += 1

                arena_results.append(
                    {
                        "sample_id": sample_id,
                        "winner": winner,
                        "reason": reason,
                        "input_question": finetuned_item["input_question"],
                    }
                )

                print(f"    Winner: {winner}")

                time.sleep(1)
                # Add delay between samples
                # if i < len(base_data) - 1:
                #    print(f"    Waiting {self.batch_delay}s before next comparison...")
                #    time.sleep(self.batch_delay)

            except Exception as e:
                import traceback

                print(traceback.format_exc())
                print(f"    ❌ Error in arena comparison for sample {sample_id}: {e}")
                arena_results.append(
                    {
                        "sample_id": sample_id,
                        "winner": "Error",
                        "reason": str(e),
                        "input_question": base_item["input_question"],
                    }
                )
                continue

        # Calculate final statistics
        total_comparisons = len([r for r in arena_results if r["winner"] != "Error"])

        arena_summary = {
            "total_comparisons": total_comparisons,
            "finetuned_wins": finetuned_wins,
            "base_wins": base_wins,
            "finetuned_win_rate": round(finetuned_wins / total_comparisons, 3)
            if total_comparisons > 0
            else 0,
            "base_win_rate": round(base_wins / total_comparisons, 3) if total_comparisons > 0 else 0,
            "overall_winner": "Finetuned"
            if finetuned_wins > base_wins
            else "Base"
            if base_wins > finetuned_wins
            else "Tie",
        }

        print("\n✓ Arena comparison complete!")
        print(
            f"  Finetuned wins: {finetuned_wins}/{total_comparisons} ({arena_summary['finetuned_win_rate']:.1%})"
        )
        print(f"  Base wins: {base_wins}/{total_comparisons} ({arena_summary['base_win_rate']:.1%})")
        print(f"  Overall winner: {arena_summary['overall_winner']}")

        return {"arena_summary": arena_summary, "detailed_results": arena_results}

    def print_arena_summary(self, arena_results: dict[str, Any]):
        """
        Print a summary of arena comparison results.

        Args:
            arena_results: Results from arena_comparison method
        """
        if "error" in arena_results:
            print(f"Arena Error: {arena_results['error']}")
            return

        summary = arena_results["arena_summary"]
        detailed = arena_results["detailed_results"]

        print("\n" + "=" * 60)
        print("ARENA COMPARISON SUMMARY")
        print("=" * 60)

        print(f"Total head-to-head comparisons: {summary['total_comparisons']}")
        print("")
        print("Results:")
        print(f"  Finetuned Model wins: {summary['finetuned_wins']} ({summary['finetuned_win_rate']:.1%})")
        print(f"  Base Model wins: {summary['base_wins']} ({summary['base_win_rate']:.1%})")
        print("")
        print(f"🏆 Overall Winner: {summary['overall_winner']} Model")

        # Show some example wins for finetuned model
        finetuned_examples = [r for r in detailed if r["winner"] == "Finetuned_Model"][:3]
        if finetuned_examples:
            print("\nExample Finetuned Model wins:")
            for i, example in enumerate(finetuned_examples, 1):
                print(f"  {i}. Sample {example['sample_id']}: {example['reason'][:100]}...")

        # Show some example wins for base model
        base_examples = [r for r in detailed if r["winner"] == "Base_Model"][:3]
        if base_examples:
            print("\nExample Base Model wins:")
            for i, example in enumerate(base_examples, 1):
                print(f"  {i}. Sample {example['sample_id']}: {example['reason'][:100]}...")

        print("\n" + "=" * 60)

    def save_results(self, results: dict[str, Any], output_path: str):
        """
        Save evaluation results to JSON file with proper formatting.

        Args:
            results: Results dictionary from compare_models
            output_path: Path to save the JSON file
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"\n✓ Results saved to: {output_path}")

            # Print file size info
            import os

            file_size = os.path.getsize(output_path)
            print(f"  File size: {file_size / 1024:.1f} KB")

        except Exception as e:
            print(f"✗ Error saving results: {e}")
            raise

    def print_summary(self, results: dict[str, Any]):
        """
        Print a simple summary of the evaluation results focused on accuracy.

        Args:
            results: Results dictionary from compare_models
        """
        stats = results["comparison_statistics"]
        metadata = results["evaluation_metadata"]

        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)

        print(f"Total samples: {metadata['total_samples']}")
        print(f"Evaluation model: {metadata['evaluation_model']}")

        # Skip if there's an error
        if "error" in stats:
            print(f"Error: {stats['error']}")
            return

        # Print results for each metric
        for metric_name, metric_stats in stats.items():
            if metric_name == "overall_summary":
                continue

            print(f"\n{metric_name}:")
            print(
                f"  Base Model      - Average: {metric_stats['base_average']:.3f}, Passed: {metric_stats['base_passed']}"
            )
            print(
                f"  Finetuned Model - Average: {metric_stats['finetuned_average']:.3f}, Passed: {metric_stats['finetuned_passed']}"
            )
            print(
                f"  Improvement     - {metric_stats['improvement']:+.3f} (Pass rate: {metric_stats['pass_rate_improvement']:+d})"
            )

        # Overall summary
        if "overall_summary" in stats:
            overall = stats["overall_summary"]
            print(f"\n{'OVERALL RESULTS'.center(30, '-')}")
            print(f"Winner: {overall['winner']} Model")
            print(f"Base Model Average: {overall['base_model_average']:.3f}")
            print(f"Finetuned Model Average: {overall['finetuned_model_average']:.3f}")
            print(f"Overall Improvement: {overall['overall_improvement']:+.3f}")

        print("\n" + "=" * 60)


# Example usage and main execution
def main():
    """
    Example usage of the RecommendationEvaluator class.
    """
    try:
        # Initialize evaluator with config from imported module
        evaluator = RecommendationEvaluator()

        # Define file paths
        base_json_path = "experiment_results/train-qwen2-recommendation-mobile/base_evaluation_outputs.json"
        finetuned_json_path = (
            "experiment_results/train-qwen2-recommendation-mobile/finetuned_evaluation_outputs.json"
        )
        output_path = "comparison_evaluation_results.json"

        # Run arena comparison

        arena_results = evaluator.arena_comparison(base_json_path, finetuned_json_path)

        evaluator.print_arena_summary(arena_results)
        evaluator.save_results(arena_results, output_path)

        print("\n✓ Evaluation completed successfully!")
        print(f"Check '{output_path}' for detailed results.")

    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
