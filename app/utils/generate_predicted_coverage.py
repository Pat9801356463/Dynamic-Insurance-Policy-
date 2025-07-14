import pandas as pd

def add_predicted_coverage_by_rule(rate_path, plan_path, output_path="Data/rate_with_coverage_final.csv"):
    rate_df = pd.read_csv(rate_path, compression="gzip", low_memory=False)
    plan_df = pd.read_csv(plan_path, compression="gzip", low_memory=False)

    merged_df = pd.merge(rate_df, plan_df[['PlanId', 'MetalLevel', 'PlanType']], on='PlanId', how='left')
    merged_df['MetalLevel'] = merged_df['MetalLevel'].fillna('Bronze')
    merged_df['PlanType'] = merged_df['PlanType'].fillna('HMO')

    if 'IndividualRate' not in merged_df.columns:
        ir_cols = [col for col in merged_df.columns if 'IndividualRate' in col]
        if ir_cols:
            merged_df['IndividualRate'] = merged_df[ir_cols].mean(axis=1)
        else:
            merged_df['IndividualRate'] = 300  # fallback

    merged_df['PredictedCoverage'] = merged_df['IndividualRate'] * 80
    merged_df['PredictedCoverage_OriginalScale'] = merged_df['PredictedCoverage'] / 80

    merged_df.to_csv(output_path, index=False)
    print(f"✅ Saved coverage-enhanced rate file at {output_path}")

    return merged_df
