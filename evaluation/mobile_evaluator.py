"""
Script for evaluating language models on mobile-specific tasks such as MiniPersonalQA and MiniRecommendation.
"""

import json
import os
import re
from tqdm import tqdm
from typing import List, Dict, Any, Union
from datasets import load_dataset, Dataset as HFDataset


class MobileEvaluator:
    def __init__(self, model, task="mini_personalqa"):
        """
        Initialize evaluator with pre-loaded model
        
        Args:
            model: Pre-loaded language model with .generate(prompt) method
            tokenizer: Optional tokenizer (not needed if model has .generate(prompt))
            device: Device to run evaluation on
        """
        self.model = model
        self.task = task
    
    def format_question_mini_personalqa(self, data_point: Dict[str, Any], few_shot_examples: List[Dict[str, Any]] = None) -> str:
        """Format question for model input with optional few-shot examples"""
        formatted = ""
        
        # Add few-shot examples if provided
        if few_shot_examples is not None:
            formatted = "Answer the multiple choice question by selecting exactly one letter: A, B, C, or D. Try to guess the correct answer.\n\n"
            for example in few_shot_examples:
                formatted += f"Question: {example['question']}\n\n"
                for choice_key, choice_value in example['choices'].items():
                    formatted += f"{choice_key}: {choice_value}\n"
                formatted += f"\n\nAnswer: {example['correct_answer']}\n\n"
        # Add the actual question
        question_text = data_point["question"]
        choices = data_point["choices"]
        
        formatted += f"Question: {question_text}\n\n"
        for choice_key, choice_value in choices.items():
            formatted += f"{choice_key}: {choice_value}\n"
        formatted += "\n\nAnswer: "
        
        return formatted
    
    def format_question_mini_recommendation(self, data_point: Dict[str, Any], few_shot_examples: List[Dict[str, Any]] = None) -> str:
        """Format recommendation question for model input with optional few-shot examples"""
        formatted = ""
        
        # Add few-shot examples if provided
        if few_shot_examples is not None:
            formatted = "Recommend best actions based on user queries.\n"
            for example in few_shot_examples:
                user_query = example['prompt']
                recommendation = example['recommendation']
                
                formatted += f"Recommend best actions based on this user query: {user_query}\n\n"
                formatted += f"Answer: {recommendation}\n\n"
        
            formatted += "Output only a single sentence answer of your recommendation.\n"
        
        # Add the actual question (without answer for inference)
        user_query = data_point["prompt"]
        
        formatted += f"Recommend best actions based on this user query: {user_query}\n\n"
        formatted += "Answer: "
        
        return formatted
    
    def predict_multichoice_answer(self, question: str) -> str:
        """Generate prediction for a single question using model's .generate() method"""
        # Use the model's built-in generate method
        prediction = self.model.generate(question)

        # Extract just the letter (A, B, C, D)
        prediction = str(prediction).upper().strip()
        for choice in ['A', 'B', 'C', 'D']:
            if choice in prediction:
                return choice
        
        return prediction  # Return full prediction if no clear choice found
    
    def predict_short_answer(self, question):
        full_response = self.model.generate(question)
        
        full_response = full_response.replace("1.", "")

        # Split by sentence endings and return first sentence
        sentences = re.split(r'[.!?]+', full_response.strip())
        
        # Return first non-empty sentence
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                return sentence
        
        # Fallback: return full response if no sentence endings found
        return full_response.strip()
    
    def evaluate(self, data_path: Union[str, List[Dict]], verbose=False, few_shot_examples: List[Dict[str, Any]] = None, save_outputs=False, save_results_dir=None) -> Dict[str, float]:
        """
        Evaluate model on the dataset
        
        Args:
            data_path: Path to JSONL file or list of data samples
            batch_size: Batch size for processing (currently supports 1)
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Load dataset using HuggingFace datasets
        if isinstance(data_path, str):
            # Load from local JSONL file
            dataset = load_dataset('json', data_files=data_path, split='train')
        else:
            # Create from list of dictionaries
            dataset = HFDataset.from_list(data_path)

        correct_predictions = 0
        total_predictions = 0
        predictions = []
        ground_truths = []
        detailed_outputs = []
        
        # Progress bar
        pbar = tqdm(dataset, desc="Evaluating", total=len(dataset))
        
        for sample in pbar:
            # Format question
            if self.task == "mini_personalqa":
                formatted_question = self.format_question_mini_personalqa(sample, few_shot_examples)
                predicted_answer = self.predict_multichoice_answer(formatted_question)
                correct_answer = sample["correct_answer"]
            elif self.task == "mini_recommendation":
                formatted_question = self.format_question_mini_recommendation(sample, few_shot_examples)
                predicted_answer = self.predict_short_answer(formatted_question)
                correct_answer = sample["recommendation"]
            else:
                raise ValueError("Task not recognized.")
            
            if save_outputs:
                output_entry = {
                    "input_question": formatted_question,
                    "predicted_answer": predicted_answer,
                    "correct_answer": correct_answer,
                    "sample_id": total_predictions
                }
                detailed_outputs.append(output_entry)
            
            # Print prompt and answers if verbose
            if verbose:
                print(f"\n{'='*80}")
                print("PROMPT:")
                print(formatted_question)
                print(f"\nPREDICTED: {predicted_answer}")
                print(f"CORRECT: {correct_answer}")
                print(f"RESULT: {'✓ CORRECT' if predicted_answer == correct_answer else '✗ INCORRECT'}")
                print('='*80)
            
            # Track results
            predictions.append(predicted_answer)
            ground_truths.append(correct_answer)
            
            # Check if correct
            is_correct = predicted_answer == correct_answer
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            # Update progress bar
            current_accuracy = correct_predictions / total_predictions * 100
            pbar.set_postfix({
                'Accuracy': f'{current_accuracy:.2f}%',
                'Correct': f'{correct_predictions}/{total_predictions}'
            })
        
        # Calculate final metrics
        accuracy = correct_predictions / total_predictions
        
        # Category-wise accuracy (if available)
        category_stats = {}
        if len(dataset) > 0 and 'category' in dataset[0]:
            categories = set(sample['category'] for sample in dataset)
            
            for category in categories:
                cat_correct = 0
                cat_total = 0
                for i, sample in enumerate(dataset):
                    if sample['category'] == category:
                        if predictions[i] == ground_truths[i]:
                            cat_correct += 1
                        cat_total += 1
                
                if cat_total > 0:
                    category_stats[category] = {
                        'accuracy': cat_correct / cat_total,
                        'correct': cat_correct,
                        'total': cat_total
                    }
        
        if save_outputs:
            # Save to JSON file
            save_output_dir = "evaluation_outputs.json"
            if save_results_dir:
                save_output_dir = os.path.join(save_results_dir, save_output_dir)
                
            with open(save_output_dir, 'w', encoding='utf-8') as f:
                json.dump(detailed_outputs, f, indent=2, ensure_ascii=False)

        results = {
            'overall_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'category_stats': category_stats,
            'predictions': predictions,
            'ground_truths': ground_truths
        }

        if save_results_dir:
            self.save_results(save_results_dir, results)

        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted evaluation results"""
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print(f"{'='*50}")
        
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
        print(f"Correct: {results['correct_predictions']}/{results['total_predictions']}")
        
        if results['category_stats']:
            print(f"\n{'Category Breakdown:':<20}")
            print(f"{'Category':<20} {'Accuracy':<10} {'Correct/Total':<15}")
            print("-" * 45)
            for category, stats in results['category_stats'].items():
                acc_pct = stats['accuracy'] * 100
                print(f"{category:<20} {acc_pct:>7.2f}% {stats['correct']:>6}/{stats['total']:<6}")
    
    def save_results(self, save_results_dir, results: Dict[str, Any]):
        """Save evaluation results to eval_results.json"""

        # Prepare per-category accuracy dictionary
        per_category_accuracy = {}
        if results['category_stats']:
            for category, stats in results['category_stats'].items():
                per_category_accuracy[category] = stats['accuracy']
        
        # Prepare data to save
        save_data = {
            "task": self.task,
            "results": results['overall_accuracy'],
            "per_category_accuracy": per_category_accuracy
        }
        
        save_path = os.path.join(save_results_dir, 'eval_results.json')

        # Save to JSON file
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {save_path}")
