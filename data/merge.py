def merge_datasets(df_main, df_external):
    df_main_t = df_main.T.reset_index().rename(columns={'index': 'question_id'})
    df_main_t['question_text_clean'] = df_main_t['question_id'].str.strip().str.lower()

    df_external['instructions_clean'] = (
        df_external['instructions'].astype(str).str.strip().str.lower()
    )

    merged = df_main_t.merge(
        df_external,
        left_on='question_text_clean',
        right_on='instructions_clean',
        how='inner'
    )

    return merged
