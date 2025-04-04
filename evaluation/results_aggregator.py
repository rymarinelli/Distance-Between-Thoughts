import pandas as pd
import numpy as np

from logic.step_parser import parse_steps
from logic.embedding import embed_steps
from logic.logic_analysis import logic_error_magnitude
from evaluation.rouge_eval import compute_rouge

response_map = {
    'response_small': 'small',
    'response_medium': 'medium',
    'response_large': 'large'
}

def collect_results(merged_df):
    """
    Processes all responses and returns a DataFrame of evaluation metrics.
    """
    records = []

    for _, row in merged_df.iterrows():
        for response_key, label in response_map.items():
            if response_key not in row:
                continue

            response = row[response_key]
            reference = row.get("solution", "")
            steps = parse_steps(response)

            rouge_l = compute_rouge(reference, response)
            if not steps:
                records.append({
                    'Question': row['question_id'],
                    'Size': label,
                    'ROUGE_L': round(rouge_l, 4),
                    'Logic_Mistakes': 0,
                    'Avg_Angle_Error': 0.0,
                    'Level': row.get('level', 'unknown'),
                    'Adversarial': row.get('adversarial', False)
                })
                continue

            embeddings = embed_steps(steps)
            errors = logic_error_magnitude(embeddings, steps)

            avg_angle = np.mean([a for _, a in errors]) if errors else 0.0

            records.append({
                'Question': row['question_id'],
                'Size': label,
                'ROUGE_L': round(rouge_l, 4),
                'Logic_Mistakes': len(errors),
                'Avg_Angle_Error': round(avg_angle, 2),
                'Level': row.get('level', 'unknown'),
                'Adversarial': row.get('adversarial', False)
            })

    return pd.DataFrame(records)
