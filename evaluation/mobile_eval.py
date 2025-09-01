from tqdm import tqdm
from typing import List, Dict, Any, Union
from datasets import load_dataset, Dataset as HFDataset

from evaluation.eval_adapter_models import CustomPeftModel

class MobileEvaluator:
    def __init__(self, model):
        """
        Initialize evaluator with pre-loaded model
        
        Args:
            model: Pre-loaded language model with .generate(prompt) method
            tokenizer: Optional tokenizer (not needed if model has .generate(prompt))
            device: Device to run evaluation on
        """
        self.model = model
    
    def format_question(self, data_point: Dict[str, Any], few_shot_examples: List[Dict[str, Any]] = None) -> str:
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
    
    def predict_answer(self, question: str) -> str:
        """Generate prediction for a single question using model's .generate() method"""
        # Use the model's built-in generate method
        prediction = self.model.generate(question)
        
        # Extract just the letter (A, B, C, D)
        prediction = str(prediction).upper().strip()
        for choice in ['A', 'B', 'C', 'D']:
            if choice in prediction:
                return choice
        
        return prediction  # Return full prediction if no clear choice found
    
    def evaluate(self, data_path: Union[str, List[Dict]], verbose=False, few_shot_examples: List[Dict[str, Any]] = None) -> Dict[str, float]:
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
        
        # Progress bar
        pbar = tqdm(dataset, desc="Evaluating", total=len(dataset))
        
        for sample in pbar:
            # Format question
            formatted_question = self.format_question(sample, few_shot_examples)
            
            # Get prediction
            predicted_answer = self.predict_answer(formatted_question)
            correct_answer = sample["correct_answer"]
            
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
        
        results = {
            'overall_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'category_stats': category_stats,
            'predictions': predictions,
            'ground_truths': ground_truths
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted evaluation results"""
        print(f"\n{'='*50}")
        print("MOBILE QA EVALUATION RESULTS")
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

# Example usage:

MINI_PERSONAL_QA_EXAMPLES = [
    {"type": "train", "category": "App Usage", "question": "What specific method is used by news apps to send me updates?", "choices": {"A": "Push notifications", "B": "Sending a letter", "C": "A town crier", "D": "Email newsletters"}, "correct_answer": "A"},
    {"type": "train", "category": "Communication & Social", "question": "The text thread with my old college buddies is always buzzing with new messages. Which of my social circles has a particularly lively group chat?", "choices": {"A": "My coworkers", "B": "My college friends", "C": "My high school acquaintances", "D": "My family"}, "correct_answer": "B"},
    {"type": "train", "category": "Location & Travel", "question": "What type of route do I use for my daily commute to my job?", "choices": {"A": "A scenic bike path", "B": "A local side street", "C": "The highway", "D": "A pedestrian walkway"}, "correct_answer": "C"}
]

def evaluate_base():
    EVAL_DATASET = "data/MiniPersonalQA_eval.jsonl"
    BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"

    model = CustomPeftModel(None, adapter_name="base", base_model=BASE_MODEL)

    model.set_generation_config(max_new_tokens=10)

    # Create evaluator - tokenizer is optional now
    evaluator = MobileEvaluator(model)

    # Run evaluation on JSONL file
    results = evaluator.evaluate(EVAL_DATASET, verbose=True, few_shot_examples=[])

    # Print results
    evaluator.print_results(results)


def evaluate_finetuned():

    ADAPTER_DIR = "experiment_results/TinyLlama_v1.1-mars-minipersonalqa/Qwen2-0.5B-mars-mini_personalqa-r8-a2"
    ADAPTER_NAME = "mars"
    EVAL_DATASET = "data/MiniPersonalQA_eval.jsonl"

    model = CustomPeftModel(ADAPTER_DIR, adapter_name=ADAPTER_NAME)

    model.set_generation_config(max_new_tokens=1)

    # Create evaluator - tokenizer is optional now
    evaluator = MobileEvaluator(model)

    # Run evaluation on JSONL file
    results = evaluator.evaluate(EVAL_DATASET, verbose=True)

    # Print results
    evaluator.print_results(results)
