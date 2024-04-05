import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1(prediction, ground_truths):
    return max([f1_score(prediction, gt) for gt in ground_truths])


if __name__ == "__main__":
    # Evaluate the F1 score of the model's answers.
    with open ("./DATA/KG/answers/hotpot_answers/tfidf_agent_responses.json", 'r') as f:
        data = json.load(f)
        
    f1_scores = []
    
    for record in data:
        gt = record['gt']
        response = record['response']
        score = f1(response, [gt])
        f1_scores.append(score)
        
    print(f"Average F1 score: {np.mean(f1_scores)}")