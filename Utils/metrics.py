import argparse

def get_target_extraction_f1(pred_file):
    gold_ents = []
    pred_ents = []
    #
    g_sent, p_sent = [], []
    g_target, p_target = [], []
    #
    for line in open(pred_file):
        if line.strip() == "":
            gold_ents.append(set(g_sent))
            pred_ents.append(set(p_sent))
            g_sent, p_sent = [], []
            g_target, p_target = [], []
        else:
            tok, gold, pred = line.strip().split("\t")
            #
            #
            if "B-targ" in gold or "I-targ" in gold:
                g_target.append(tok)
            else:
                if len(g_target) > 0:
                    g_sent.append(" ".join(g_target))
                    g_target = []
            #
            #
            if "B-targ" in gold or "I-targ" in pred:
                p_target.append(tok)
            else:
                if len(p_target) > 0:
                    p_sent.append(" ".join(p_target))
                    p_target = []
                    #
    tp = 0
    gold_len = 0
    pred_len = 0
    for gsent, psent in zip(gold_ents, pred_ents):
        tp += len(gsent.intersection(psent))
        gold_len += len(gsent)
        pred_len += len(psent)
        #
    prec = tp / gold_len
    rec = tp / pred_len
    f1 = 2 * prec * rec / (prec + rec + 0.000001)
    return prec, rec, f1, gold_ents, pred_ents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")

    args = parser.parse_args()
    prec, rec, f1, gold_ents, pred_ents = get_target_extraction_f1(args.file)
    print("Prec: {0:.3f}".format(prec))
    print("Rec: {0:.3f}".format(rec))
    print("F1: {0:.3f}".format(f1))
