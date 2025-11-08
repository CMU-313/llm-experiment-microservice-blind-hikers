"""Evaluation helpers for the Basic LLM experiment workflow.

This module collects the translation and language-detection evaluation sets
used in the prototype notebooks, along with simple scoring utilities that can
be re-used in automated scripts or ad-hoc experiments.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Mapping, MutableMapping


translation_eval_set = [
    {
        "post": "Hier ist dein erstes Beispiel.",
        "expected_answer": "Here is your first example.",
    },
    {
        "post": "¿Cómo te sientes hoy?",
        "expected_answer": "How do you feel today?",
    },
    {
        "post": "Je voudrais une tasse de café, s'il vous plaît.",
        "expected_answer": "I would like a cup of coffee, please.",
    },
    {
        "post": "今日はとても暑いです。",
        "expected_answer": "It is very hot today.",
    },
    {
        "post": "Buongiorno! Come va la tua giornata?",
        "expected_answer": "Good morning! How is your day going?",
    },
    {
        "post": "나는 음악을 듣는 것을 좋아해요.",
        "expected_answer": "I like listening to music.",
    },
    {
        "post": "Спасибо за помощь вчера.",
        "expected_answer": "Thank you for the help yesterday.",
    },
    {
        "post": "Où est la gare la plus proche?",
        "expected_answer": "Where is the nearest train station?",
    },
    {
        "post": "这本书非常有趣。",
        "expected_answer": "This book is very interesting.",
    },
    {
        "post": "هل يمكنك مساعدتي من فضلك؟",
        "expected_answer": "Can you help me, please?",
    },
]


language_detection_eval_set = [
    {
        "post": "Hier ist dein erstes Beispiel.",
        "expected_answer": "German",
    },
    {
        "post": "¿Cómo te sientes hoy?",
        "expected_answer": "Spanish",
    },
    {
        "post": "Je voudrais une tasse de café, s'il vous plaît.",
        "expected_answer": "French",
    },
    {
        "post": "今日はとても暑いです。",
        "expected_answer": "Japanese",
    },
    {
        "post": "Buongiorno! Come va la tua giornata?",
        "expected_answer": "Italian",
    },
    {
        "post": "나는 음악을 듣는 것을 좋아해요.",
        "expected_answer": "Korean",
    },
    {
        "post": "Спасибо за помощь вчера.",
        "expected_answer": "Russian",
    },
    {
        "post": "Où est la gare la plus proche?",
        "expected_answer": "French",
    },
    {
        "post": "这本书非常有趣。",
        "expected_answer": "Chinese (Simplified)",
    },
    {
        "post": "هل يمكنك مساعدتي من فضلك؟",
        "expected_answer": "Arabic",
    },
]


def normalize(text: str) -> str:
    """Normalize text for lenient comparison by stripping punctuation/case."""

    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def eval_single_response_classification(expected_answer: str, llm_response: str) -> float:
    """Return 1.0 for an exact match (after normalization), else 0.0."""

    return 1.0 if normalize(expected_answer) == normalize(llm_response) else 0.0


def evaluate(
    query_fn: Callable[[str], str],
    eval_fn: Callable[[str, str], float],
    dataset: Iterable[Mapping[str, str] | MutableMapping[str, str]],
) -> float:
    """Compute the average evaluation score across the dataset."""

    total_score = 0.0
    count = 0
    for sample in dataset:
        post = sample["post"]
        expected = sample["expected_answer"]

        llm_response = query_fn(post)
        score = eval_fn(expected, llm_response)
        total_score += score
        count += 1

        print("\nPost:", post)
        print("Expected:", expected)
        print("LLM Response:", llm_response)
        print(f"Score: {score:.3f}")

    if count == 0:
        raise ValueError("Dataset is empty; provide at least one evaluation sample")

    return total_score / count


__all__ = [
    "translation_eval_set",
    "language_detection_eval_set",
    "normalize",
    "eval_single_response_classification",
    "evaluate",
]


