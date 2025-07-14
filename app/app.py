# app.py

import streamlit as st
import pandas as pd
import os

from utils.plan_matcher import (
    match_plans_with_coverage,
    show_top_unique_plans,
    enrich_with_benefits,
    explain_top_plans,
)
from utils.rules_engine import RegulatorySearchEngine
from chatbot import InsuranceChatbot
from utils.generate_predicted_coverage import add_predicted_coverage_by_rule

# === Load & Generate Data ===
# === Load & Generate Data ===
@st.cache_data
def load_data():
    try:
        # ✅ Define all paths inside the function
        rate_path = "Data/rate-puf.csv.gz"
        plan_path = "Data/plan_df.csv.gz"
        output_path = "/tmp/rate_with_coverage_final.csv"  # safe for Streamlit Cloud

        st.info("🔍 Reading source CSVs...")
        plan_df = pd.read_csv(plan_path, compression="gzip", low_memory=False)
        rate_df_raw = pd.read_csv(rate_path, compression="gzip", low_memory=False)
        benefits_df = pd.read_csv("Data/benefits_df.csv.gz", compression="gzip", low_memory=False)

        st.info("⚙️ Generating predicted coverage...")
        rate_df = add_predicted_coverage_by_rule(
            rate_path=rate_path,
            plan_path=plan_path,
            output_path=output_path
        )

        return plan_df, rate_df, benefits_df

    except Exception as e:
        st.error(f"❌ Failed to load or process data: {e}")
        st.stop()


plan_df, rate_df, benefits_df = load_data()

# === Initialize Engines ===
reg_engine = RegulatorySearchEngine(
    embedding_path="Data/legal_doc_embeddings.npy",
    metadata_path="Data/legal_docs_metadata.json"
)
chatbot = InsuranceChatbot(
    embedding_path="Data/legal_doc_embeddings.npy",
    metadata_path="Data/legal_docs_metadata.json"
)

# === Streamlit UI ===
st.title("🏥 Dynamic Insurance Policy Matcher + Advisor")

with st.form("user_input_form"):
    st.subheader("Enter Your Insurance Requirements")

    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    valid_state_codes = sorted([
        'AK', 'AL', 'AZ', 'FL', 'IN', 'LA', 'MO', 'MS', 'NC', 'ND', 'OK', 'SC',
        'TN', 'TX', 'WI', 'WY', 'AR', 'DE', 'HI', 'IA', 'IL', 'KS', 'MI', 'MT',
        'NE', 'NH', 'OH', 'OR', 'SD', 'UT', 'WV'
    ])
    state_code = st.selectbox("State Code", options=valid_state_codes, index=valid_state_codes.index("TX"))

    target_coverage = st.number_input("Target Coverage Amount ($)", min_value=1000, value=30000)

    plan_type = st.selectbox("Preferred Plan Type", options=["Any", "HMO", "PPO", "EPO"], index=0)

    submitted = st.form_submit_button("🔍 Find Matching Plans")

# Step 2: Match and show plans
if submitted:
    st.subheader("📋 Top Matching Plans")

    matched = match_plans_with_coverage(
        age=age,
        state_code=state_code,
        target_coverage=target_coverage,
        plan_df=plan_df,
        rate_df=rate_df,
        plan_type=None if plan_type == "Any" else plan_type
    )

    if matched.empty:
        st.warning("No plans found matching your criteria.")
    else:
        top = show_top_unique_plans(matched, top_n=20)
        enriched = enrich_with_benefits(top, benefits_df)
        st.dataframe(enriched[[
            "PlanId", "PlanMarketingName", "PlanType",
            "IndividualRate", "PredictedCoverage", "CoveredBenefits"
        ]])

        explanations = explain_top_plans(enriched, age=age)
        for i, exp in enumerate(explanations):
            st.markdown(f"#### Plan #{i+1}")
            st.markdown(exp)

        # Step 3: Disclaimer
        st.info("📌 **Disclaimer**: Final rates and coverage may vary based on full identity details and company updates. Please verify on the official insurer website.")

        # Step 4: Regulation Summary
        st.subheader("📘 Key Regulations You Should Know")
        rule_query = f"What are the important insurance rules for age {age}, state {state_code}, plan type {plan_type}?"
        rule_docs = reg_engine.search(rule_query, top_k=1)
        if rule_docs:
            st.markdown(rule_docs[0].page_content[:1000] + "...")
        else:
            st.warning("No relevant regulation info found.")

        # Step 5: Chatbot
        st.subheader("💬 Ask a Follow-Up Question")
        user_query = st.text_input("Your question:")
        if user_query:
            response = chatbot.respond(user_query)
            st.markdown(response)
