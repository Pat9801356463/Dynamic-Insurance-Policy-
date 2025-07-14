# chatbot.py

import re
from rules_engine import RegulatorySearchEngine

class InsuranceChatbot:
    def __init__(self, embedding_path, metadata_path):
        self.engine = RegulatorySearchEngine(embedding_path, metadata_path)
        self.policy_keywords = [
            "plan id", "planid", "specific plan", "premium of plan", "details of plan",
            "this plan", "policy", "this insurance", "coverage of plan", "plan benefits",
            "compare plan", "what is the price of plan", "cost of plan"
        ]

    def is_policy_specific(self, query: str) -> bool:
        return any(re.search(rf"\b{re.escape(kw)}\b", query.lower()) for kw in self.policy_keywords)

    def respond(self, query: str) -> str:
        if self.is_policy_specific(query):
            return (
                "🔍 For detailed questions about a specific insurance policy, "
                "please visit the respective insurance provider's official website for the latest information."
            )
        else:
            docs = self.engine.search(query)
            if not docs:
                return "❌ Sorry, I couldn't find any relevant information on that topic."

            top_doc = docs[0]
            return f"📘 **Answer from regulatory document**:\n\n{top_doc.page_content.strip()[:1000]}..."
