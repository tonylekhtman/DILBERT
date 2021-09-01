def tag2aspect(tag_sequence):
    """
    convert BIO tag sequence to the aspect sequence
    :param tag_sequence: tag sequence in BIO tagging schema
    :return:
    """
    ts_sequence = []
    beg = -1
    for index, ts_tag in enumerate(tag_sequence):
        if ts_tag == 'O':
            if beg != -1:
                ts_sequence.append((beg, index-1))
                beg = -1
        else:
            cur = ts_tag.split('-')[0]  # unified tags
            if cur == 'B':
                if beg != -1:
                    ts_sequence.append((beg, index-1))
                beg = index

    if beg != -1:
        ts_sequence.append((beg, index))
    return ts_sequence

def match(pred, gold):
    true_count = 0
    for t in pred:
        if t in gold:
            true_count += 1
    return true_count

def evaluate_chunk(test_Y, pred_Y):
    """
    evaluate function for aspect term extraction
    :param test_Y: gold standard tags (i.e., post-processed labels)
    :param pred_Y: predicted tags
    :return:
    """
    assert len(test_Y) == len(pred_Y)
    length = len(test_Y)
    TP, FN, FP = 0, 0, 0

    for i in range(length):
        gold = test_Y[i]
        pred = pred_Y[i]
        assert len(gold) == len(pred)
        gold_aspects = tag2aspect(gold)
        pred_aspects = tag2aspect(pred)
        n_hit = match(pred=pred_aspects, gold=gold_aspects)
        TP += n_hit
        FP += (len(pred_aspects) - n_hit)
        FN += (len(gold_aspects) - n_hit)
    precision = float(TP) / float(TP + FP + 0.00001)
    recall = float(TP) / float(TP + FN + 0.0001)
    F1 = 2 * precision * recall / (precision + recall + 0.00001)
    return precision, recall, F1