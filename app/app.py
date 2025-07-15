# app.py

from flask import Flask, render_template, request
import pandas as pd
import os

from utils.plan_matcher import (
    match_plans_with_coverage,
    show_top_unique_plans,
    enrich_with_benefits,
    explain_top_plans
)
from utils.rules_engine import RegulatorySearchEngine
from utils.generate_predicted_coverage import add_predicted_coverage_by_rule
from chatbot import InsuranceChatbot

app = Flask(__name__)

# === Load data once at startup ===
try:
    print("🔍 Loading data...")
    rate_path = "Data/rate-puf.csv.gz"
    plan_path = "Data/plan_df.csv.gz"

    plan_df = pd.read_csv(plan_path, compression="gzip", low_memory=False)
    rate_df = add_predicted_coverage_by_rule(
        rate_path=rate_path,
        plan_path=plan_path,
        output_path=None  # In-memory only
    )
    benefits_df = pd.read_csv("Data/benefits_df.csv.gz", compression="gzip", low_memory=False)

    reg_engine = RegulatorySearchEngine(
        embedding_path="Data/legal_doc_embeddings.npy",
        metadata_path="Data/legal_docs_metadata.json"
    )
    chatbot = InsuranceChatbot(
        embedding_path="Data/legal_doc_embeddings.npy",
        metadata_path="Data/legal_docs_metadata.json"
    )
    print("✅ Data loaded successfully.")

except Exception as e:
    print(f"❌ Failed to load data: {e}")
    plan_df = rate_df = benefits_df = None
    chatbot = reg_engine = None

# === Homepage ===
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get form input
            age = int(request.form["age"])
            state_code = request.form["state_code"]
            plan_type = request.form["plan_type"]
            target_coverage = float(request.form["target_coverage"])

            # Match plans
            matched = match_plans_with_coverage(
                age=age,
                state_code=state_code,
                target_coverage=target_coverage,
                plan_df=plan_df,
                rate_df=rate_df,
                plan_type=None if plan_type == "Any" else plan_type
            )

            if matched.empty:
                return render_template("index.html", error="No matching plans found.")

            # Top plans + benefits
            top = show_top_unique_plans(matched, top_n=20)
            enriched = enrich_with_benefits(top, benefits_df)
            explanations = explain_top_plans(enriched, age)

            # Regulation info
            rule_query = f"What are the important insurance rules for age {age}, state {state_code}, plan type {plan_type}?"
            rule_docs = reg_engine.search(rule_query, top_k=1)
            rules = rule_docs[0].page_content[:1000] + "..." if rule_docs else "No regulation found."

            return render_template(
                "index.html",
                plans=enriched.to_dict(orient="records"),
                explanations=explanations,
                rules=rules
            )

        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")

    # GET request
    return render_template("index.html", plans=None)

# === API for chatbot (optional AJAX) ===
@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.form.get("query")
    if chatbot and user_query:
        response = chatbot.respond(user_query)
        return {"response": response}
    return {"response": "Chatbot unavailable."}

if __name__ == "__main__":
    app.run(debug=True)
