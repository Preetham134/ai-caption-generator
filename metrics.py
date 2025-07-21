from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams
import numpy as np

def distinct_n(captions, n=2):
    ngrams_set = set()
    total = 0
    for cap in captions:
        tokens = cap.split()
        ng = list(ngrams(tokens, n))
        total += len(ng)
        ngrams_set.update(ng)
    return len(ngrams_set) / total if total else 0

def self_bleu(captions):
    scores = []
    for i in range(len(captions)):
        ref = captions[i]
        others = captions[:i] + captions[i+1:]
        scores.append(sentence_bleu([o.split() for o in others], ref.split()))
    return np.mean(scores)