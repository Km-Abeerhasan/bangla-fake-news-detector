"""
src/retrieval/bm25_search.py
Deterministic BM25 Sparse Lexical Retriever customized for Bengali News Fact-Checking.
Implements Okapi BM25 scoring with Bengali stop-word pruning and token normalization.
"""

import math
import re
from typing import List, Dict, Tuple
from collections import Counter


BENGALI_STOPWORDS = {
    "এবং", "ও", "করে", "করা", "হয়েছে", "হয়", "আছে", "ছিল", "জন্য",
    "হতে", "থেকে", "দিয়ে", "বা", "কিন্তু", "যদি", "তবে", "তার", "এই",
    "যে", "সে", "কি", "কী", "না", "নি", "এক", "একটি", "বলে", "জানায়"
}


def tokenize_bengali(text: str) -> List[str]:
    """
    Cleans punctuation and extracts Bengali word tokens, filtering out stop-words.
    """
    # Keep Bengali Unicode characters (0980-09FF) and standard alphanumeric tokens
    tokens = re.findall(r"[\u0980-\u09FFa-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in BENGALI_STOPWORDS and len(t) > 1]


class BengaliBM25Searcher:
    """
    In-memory Okapi BM25 index over retrieved passage dockets.
    """

    def __init__(self, passages: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = list(passages.keys())
        self.corpus_size = len(self.doc_ids)
        self.doc_lens: Dict[str, int] = {}
        self.doc_freqs: Dict[str, Counter] = {}
        self.df: Counter = Counter()

        for doc_id, text in passages.items():
            tokens = tokenize_bengali(text)
            self.doc_lens[doc_id] = len(tokens)
            counts = Counter(tokens)
            self.doc_freqs[doc_id] = counts
            for token in counts.keys():
                self.df[token] += 1

        self.avg_doc_len = (
            sum(self.doc_lens.values()) / self.corpus_size if self.corpus_size > 0 else 1.0
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Calculates Okapi BM25 score for the query against all indexed passages.
        Returns sorted list of (passage_id, score).
        """
        query_tokens = tokenize_bengali(query)
        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in self.doc_ids}

        for token in query_tokens:
            if token not in self.df:
                continue

            # Inverse Document Frequency (IDF) with smoothing
            df_val = self.df[token]
            idf = math.log(1.0 + (self.corpus_size - df_val + 0.5) / (df_val + 0.5))

            for doc_id in self.doc_ids:
                tf = self.doc_freqs[doc_id].get(token, 0)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[doc_id] / self.avg_doc_len))
                    scores[doc_id] += idf * ((tf * (self.k1 + 1.0)) / denom)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
