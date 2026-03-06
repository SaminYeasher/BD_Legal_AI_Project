from bert_score import score   


def bertscore_similarity(true_answer, predicted_answer):
    """
    Computes BERTScore similarity between
    ground truth and predicted answer.
    Works for Bangla + English.
    """

    P, R, F1 = score(
        [predicted_answer],   # Candidate
        [true_answer],        # Reference
        #model_type="bert-base-multilingual-cased",
        model_type="distilbert-base-multilingual-cased",
        verbose=False
    )

    return round(F1.mean().item(), 3)