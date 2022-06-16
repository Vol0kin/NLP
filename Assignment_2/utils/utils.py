
def evaluate_corpus(sequences, sequences_predictions, ignore_tag_code=-1):
    """Evaluate classification accuracy at corpus level, comparing with
    gold standard.
    
    Parameter ignore_tag_code (optional):
        Provides a tag code that does not count for the accuracy (used for instance with the tag 'O')
    """
    total = 0.0
    correct = 0.0
    for i, sequence in enumerate(sequences):
        pred = sequences_predictions[i]
        for j, y_hat in enumerate(pred.y):
            if sequence.y[j] != ignore_tag_code:  # The code ignore_tag_code does not count for accuracy
                if sequence.y[j] == y_hat:
                    correct += 1
                total += 1
    return correct / total
