# rules_engine.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

class RegulatorySearchEngine:
    def __init__(self, embedding_path, metadata_path, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = np.load(embedding_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)
        self.docs = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in doc_data
        ]

    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])[0]
        similarities = np.dot(self.doc_embeddings, query_embedding) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.docs[i] for i in top_indices]
