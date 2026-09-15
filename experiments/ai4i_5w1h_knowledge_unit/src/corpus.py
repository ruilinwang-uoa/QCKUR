"""Domain knowledge corpus for the RAG baselines (B6/B7).

The corpus holds AI4I documentation, a maintenance manual, and statistical
interpretation references as unstructured text. It deliberately EXCLUDES the
framework's answer templates, so the B6/B7 vs B8 contrast concerns the *form of
knowledge organization* (unstructured retrieval vs structured knowledge units),
not access to the answers.

Default retrieval is lexical (sklearn TF-IDF + cosine); if faiss-cpu and an
embedding endpoint are available it upgrades to dense retrieval.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config as C
import llm_config


def build_corpus() -> list[dict]:
    docs: list[dict] = []

    def add(source, title, text):
        docs.append({"id": f"d{len(docs)+1:03d}", "source": source,
                     "title": title, "text": text})

    # --- AI4I dataset documentation ---
    add("ai4i_doc", "Dataset overview",
        "The AI4I 2020 Predictive Maintenance Dataset contains 10,000 synthetic but "
        "physically realistic milling machine records. The binary target indicates "
        "machine failure; the failure rate is about 3.4%, making it a highly imbalanced "
        "classification problem. Each record has product type (L/M/H), air temperature, "
        "process temperature, rotational speed, torque, and tool wear.")
    add("ai4i_doc", "Features and units",
        "Air temperature and process temperature are in Kelvin. Rotational speed is in "
        "rpm, torque in Nm, tool wear in minutes. Mechanical power in Watts equals torque "
        "times rotational speed times 2*pi/60. Temperature difference is process minus air.")
    add("ai4i_doc", "Failure mode TWF",
        "Tool Wear Failure (TWF): the tool fails or is replaced when tool wear reaches "
        "200 to 240 minutes; at 240 minutes the tool is certain to fail.")
    add("ai4i_doc", "Failure mode HDF",
        "Heat Dissipation Failure (HDF): occurs when the process-air temperature "
        "difference drops below 8.6 K while rotational speed is below 1380 rpm, "
        "indicating insufficient cooling.")
    add("ai4i_doc", "Failure mode PWF",
        "Power Failure (PWF): occurs when mechanical power falls outside the safe "
        "operating envelope of 3500 to 9000 Watts.")
    add("ai4i_doc", "Failure mode OSF",
        "Overstrain Failure (OSF): occurs when the product of tool wear and torque "
        "exceeds the type-specific limit (L: 11000, M: 12000, H: 13000 min*Nm).")
    add("ai4i_doc", "Failure mode RNF",
        "Random Failure (RNF): a stochastic failure independent of the process "
        "variables; it cannot be predicted deterministically from the features.")

    # --- Maintenance manual ---
    add("manual", "General inspection",
        "Routine condition monitoring and tool-life tracking reduce unplanned stoppages. "
        "Prioritize alarms by severity and route them to the correct responder.")
    add("manual", "TWF handling",
        "On a tool wear failure or when wear enters the 200-240 min band, schedule tool "
        "replacement. The cutting tool is the responsible component; the operator handles it.")
    add("manual", "HDF handling",
        "On heat dissipation risk, inspect the cooling system, hydraulic circuit, and "
        "ventilation; clean or replace filters. This is a high-severity condition handled "
        "by the maintenance team.")
    add("manual", "PWF handling",
        "On power failure, reduce feed rate and torque to bring power within the 3500-9000 W "
        "band; check motor and drive load. High severity; operator and maintenance respond.")
    add("manual", "OSF handling",
        "On overstrain, reduce the tool-wear-times-torque load below the type limit, e.g., "
        "lower feed rate or rebalance the tool load. Medium severity; operator handles it.")
    add("manual", "RNF handling",
        "Random faults are logged and the machine inspected; no deterministic corrective "
        "action applies. Low severity; operator monitors.")

    # --- Statistical interpretation references ---
    add("stats_ref", "Imbalanced classification",
        "Under severe class imbalance, accuracy is misleading because the majority class "
        "dominates. Use F1-macro and the area under the precision-recall curve (AUPRC) "
        "instead; AUPRC is sensitive to performance on the minority (failure) class.")
    add("stats_ref", "Feature importance",
        "Tree ensembles such as random forest and gradient boosting provide feature "
        "importances that quantify each variable's contribution to the failure prediction. "
        "Torque, rotational speed, and tool wear are typically the strongest drivers.")
    add("stats_ref", "Failure detection",
        "A reliable failure detector should balance precision (few false alarms) and recall "
        "(few missed failures). In predictive maintenance, missing a failure is usually far "
        "more costly than a false alarm, so recall is weighted heavily.")
    add("stats_ref", "Correlation and causation",
        "Correlation between a sensor and the failure label is not causation. Physical rules "
        "(power envelope, thermal difference, overstrain) give the causal explanation that "
        "supports maintenance decisions.")

    return docs


class Retriever:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.vec = TfidfVectorizer(ngram_range=(1, 2)).fit([d["text"] for d in docs])
        self.mat = self.vec.transform([d["text"] for d in docs])

    def query(self, question: str, k: int = 5) -> list[dict]:
        q = self.vec.transform([question])
        sims = cosine_similarity(q, self.mat).ravel()
        idx = np.argsort(sims)[::-1][:k]
        return [{"id": self.docs[i]["id"], "source": self.docs[i]["source"],
                 "title": self.docs[i]["title"], "text": self.docs[i]["text"],
                 "score": round(float(sims[i]), 4)} for i in idx]


def get_corpus_and_retriever():
    path = C.KNOWLEDGE_DIR / "rag_corpus.json"
    if not path.exists():
        docs = build_corpus()
        path.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        docs = json.loads(path.read_text(encoding="utf-8"))
    return docs, Retriever(docs)


if __name__ == "__main__":
    docs, ret = get_corpus_and_retriever()
    print(f"corpus: {len(docs)} docs, sources:",
          sorted(set(d["source"] for d in docs)))
    embedding = llm_config.embedding_model()
    print("embedding_model:", embedding or "(disabled) -> using TF-IDF retrieval")
    for q in ["Why does power failure happen?",
              "What does AUPRC measure?",
              "How to handle heat dissipation failure?"]:
        top = ret.query(q, k=2)
        print(f"\nQ: {q}")
        for t in top:
            print(f"  [{t['score']}] {t['source']}/{t['title']}")
