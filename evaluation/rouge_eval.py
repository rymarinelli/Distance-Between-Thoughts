from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def compute_rouge(reference: str, candidate: str):
    """
    Computes the ROUGE-L f-measure score between reference and candidate.
    """
    if not reference or not candidate:
        return 0.0
    score = scorer.score(reference, candidate)
    return score['rougeL'].fmeasure
