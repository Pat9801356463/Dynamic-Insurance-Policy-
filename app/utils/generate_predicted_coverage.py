# generate_predicted_coverage.py

import pandas as pd

def add_predicted_coverage_by_rule(rate_path, plan_path, output_path="rate_with_coverage_final.csv"):
    """
    Generates 'PredictedCoverage' using a fixed multiplier rule on IndividualRate.
    Outputs a CSV ready for plan matching.
    """
    rate_df = pd.read_csv(rate_path)
    plan_df = pd.read_csv(plan_path)

    # Merge required plan details
    merged_df = pd.merge(rate_df, plan_df[['PlanId', 'MetalLevel', 'PlanType']], on='PlanId', how='left')

    # Default fill
    merged_df['MetalLevel'] = merged_df['MetalLevel'].fillna('Bronze')
    merged_df['PlanType'] = merged_df['PlanType'].fillna('HMO')

    # Ensure IndividualRate column exists
    if 'IndividualRate' not in merged_df.columns:
        rate_cols = [col for col in merged_df.columns if 'IndividualRate' in col]
        if rate_cols:
            merged_df['IndividualRate'] = merged_df[rate_cols].mean(axis=1)
        else:
            merged_df['IndividualRate'] = 300  # fallback default

    # 🔥 Rule-based predicted coverage
    merged_df['PredictedCoverage'] = merged_df['IndividualRate'] * 80
    merged_df['PredictedCoverage_OriginalScale'] = merged_df['PredictedCoverage'] / 80  # optional

  
    return merged_df

# Optional CLI usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate predicted coverage CSV using rule.")
    parser.add_argument("--rate", required=True, help="Path to rate CSV (e.g., rate.csv)")
    parser.add_argument("--plan", required=True, help="Path to plan CSV (e.g., plan_df.csv)")
    parser.add_argument("--output", default="rate_with_coverage_final.csv", help="Path to output CSV")

    args = parser.parse_args()

    add_predicted_coverage_by_rule(args.rate, args.plan, args.output)
