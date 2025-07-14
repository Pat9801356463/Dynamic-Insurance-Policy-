# plan_matcher.py

import pandas as pd

def match_plans_with_coverage(age, state_code, target_coverage, plan_df, rate_df, plan_type=None, tolerance_pct=10):
    plan_df['BasePlanId'] = plan_df['PlanId'].str.extract(r'(^\d+[A-Z]{2}\d{7})')
    rate_df['PlanId'] = rate_df['PlanId'].astype(str)
    plan_filtered = plan_df[plan_df['StateCode'].str.upper() == state_code.upper()]
    if plan_type:
        plan_filtered = plan_filtered[plan_filtered['PlanType'].str.lower() == plan_type.lower()]
    rate_filtered = rate_df[rate_df['Age'] == str(age)].copy()
    merged = plan_filtered.merge(rate_filtered, left_on='BasePlanId', right_on='PlanId', how='inner')
    lower_bound = target_coverage * (1 - tolerance_pct / 100)
    upper_bound = target_coverage * (1 + tolerance_pct / 100)
    filtered = merged[(merged['PredictedCoverage'] >= lower_bound) & (merged['PredictedCoverage'] <= upper_bound)]
    return filtered[[
        'PlanId_x', 'PlanMarketingName', 'PlanType_x', 'Age',
        'IndividualRate', 'PredictedCoverage', 'StandardComponentId'
    ]].rename(columns={
        'PlanId_x': 'PlanId',
        'PlanType_x': 'PlanType'
    }).drop_duplicates()

def show_top_unique_plans(matched_plans, top_n=20):
    if 'StandardComponentId' in matched_plans.columns:
        grouped = matched_plans.sort_values('IndividualRate').groupby('StandardComponentId', as_index=False).first()
        return grouped.sort_values('IndividualRate').head(top_n)
    else:
        deduped = matched_plans.sort_values('IndividualRate').drop_duplicates(
            subset=['PlanMarketingName', 'PlanType', 'IndividualRate'])
        return deduped.head(top_n)

def enrich_with_benefits(top_plans_df, benefits_df):
    if 'StandardComponentId' not in top_plans_df.columns:
        top_plans_df['CoveredBenefits'] = 'N/A'
        return top_plans_df
    benefits_filtered = benefits_df[benefits_df['StandardComponentId'].isin(top_plans_df['StandardComponentId'])]
    benefit_summary = (
        benefits_filtered[benefits_filtered['IsCovered'] == 'Yes']
        .groupby('StandardComponentId')['BenefitName']
        .apply(lambda x: ', '.join(sorted(set(x.dropna()))))
        .reset_index()
        .rename(columns={'BenefitName': 'CoveredBenefits'})
    )
    enriched = top_plans_df.merge(benefit_summary, on='StandardComponentId', how='left')
    enriched['CoveredBenefits'] = enriched['CoveredBenefits'].fillna('N/A')
    return enriched

def explain_top_plans(enriched_df, age):
    explanations = []
    for _, row in enriched_df.iterrows():
        explanation = f"""
🔹 **Plan ID:** {row.get('PlanId')}
🔹 **Plan Name:** {row.get('PlanMarketingName')} ({row.get('PlanType')})
🔹 **Premium:** ${row.get('IndividualRate')} for age {age}
🔹 **Predicted Coverage:** ${row.get('PredictedCoverage'):,.2f}
🔹 **Why this plan?**
   - It is among the most affordable options for your age
   - Offered as a {row.get('PlanType')} plan (consider network/referral rules)
   - Covers major services such as: {row.get('CoveredBenefits') if isinstance(row.get('CoveredBenefits'), str) else 'N/A'}
        """.strip()
        explanations.append(explanation)
    return explanations
