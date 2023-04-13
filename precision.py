def basicPrecision(true_labels, predicted_labels):
    n_samples = len(true_labels)
    true_positives = 0
    
    for i in range(n_samples):
        if true_labels[i] == predicted_labels[i]:
            true_positives += 1
    
    return true_positives / len(predicted_labels)

