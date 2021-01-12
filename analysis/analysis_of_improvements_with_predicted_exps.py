import json
import argparse
from sklearn.metrics import f1_score
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling_type", default="cls", help="cls, first, max, max-min-pooled, pooled ")
    parser.add_argument("--exp_model", default="darmstadt/services")

    args = parser.parse_args()

    # get the targets, texts, golds and predictions for all five runs WITHOUT augmenting the input
    original = []
    for run in "1 2 3 4 5".split():
        with open("../classification_results/targets_polarity/wang/{0}/{1}/test_preds.json".format(args.pooling_type, run)) as o:
            original.append(json.load(o))

    # get the targets, texts, golds and predictions for all five runs WITH augmented input
    augmented = []
    for run in "1 2 3 4 5".split():
        with open("../classification_results/predicted_data/wang/targets_polarity/{0}/{1}/{2}/test_preds.json".format(args.exp_model, args.pooling_type, run)) as o:
            augmented.append(json.load(o))


    # check that they are the same length and have the same targets
    assert original[0]["targets"] == augmented[0]["targets"]
    assert original[0]["golds"] == augmented[0]["golds"]


    # keep the targets, texts that are predicted INCORRECTLY by original and CORRECTLY by augmented on ALL 5 runs
    correct = []

    for i in range(len(original[0]["predictions"])):
        original_incorrect_augmented_correct = 0
        for run in [0, 1, 2, 3, 4]:
            if (original[run]["predictions"][i] != original[run]["golds"][i]) and (augmented[run]["predictions"][i] == augmented[run]["golds"][i]):
                original_incorrect_augmented_correct += 1
        if original_incorrect_augmented_correct > 2:
            correct.append((original[0]["predictions"][i],
                            augmented[0]["predictions"][i],
                            augmented[0]["golds"][i],
                            augmented[0]["targets"][i],
                            augmented[0]["texts"][i]))

    original_scores = []
    augmented_scores = []
    for run in [0, 1, 2, 3, 4]:
        original_scores.append(f1_score(original[run]["golds"],
                                        original[run]["predictions"],
                                        labels=[0, 1, 2],
                                        average="macro"))
        augmented_scores.append(f1_score(augmented[run]["golds"],
                                         augmented[run]["predictions"],
                                         labels=[0, 1, 2],
                                         average="macro"))


    for n in correct:
        for t in n:
            print(t)
        print("-" * 80)
        print()

    print("original F1: {0:.3f}".format(np.array(original_scores).mean()))
    print("augmented F1: {0:.3f}".format(np.array(augmented_scores).mean()))
    print()
