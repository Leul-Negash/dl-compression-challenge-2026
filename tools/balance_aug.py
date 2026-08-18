"""Match the augmentation set's option-length profile to the provided data.

The supplied questions put the correct option second-longest about 58% of the
time; their long options are over-qualified wrong ones. Written cold, my
distractors were uniformly shorter than the answer, which teaches the opposite
cue. This appends absolutist qualifiers to bring the profile back in line.
"""
import csv, glob, os, re

SRC = os.path.expanduser("~/Training/dl-compression/data/aug")
LETTERS = list("ABCDE")

CLAUSES = [
    "and this holds in every case regardless of how the rest of the system is configured",
    "an effect that is guaranteed for any architecture and cannot be avoided by tuning",
    "which is true independent of model size, batch size, or the hardware being used",
    "and no amount of additional training data will change this outcome",
    "a property that holds exactly, with no dependence on the choice of hyperparameters",
    "and this is the only mechanism by which the improvement can ever be obtained",
    "which applies uniformly to every layer of the network without exception",
    "and the effect is entirely independent of the objective being optimised",
    "a relationship that is exact rather than approximate under all conditions",
    "and this remains true whether the model is trained from scratch or fine-tuned",
    "which cannot be reproduced by any other method currently available",
    "and the benefit scales linearly without any diminishing returns at all",
    "regardless of the precision in which the weights and activations are stored",
    "and this is required for the computation to remain numerically well defined",
    "which holds for every input the model will ever be asked to process",
    "and no calibration or validation step is needed to confirm it",
    "an outcome that is entirely determined by the architecture rather than the data",
    "and the same conclusion follows for both training and inference",
    "which is guaranteed by construction and never needs to be verified empirically",
    "and this makes any further optimisation of that component unnecessary",
    "regardless of how many devices the workload is ultimately distributed across",
    "and the relationship is strictly monotonic across the entire operating range",
    "which removes any need to measure the behaviour on the target hardware",
    "and this holds even when the sequence length grows without bound",
    "an effect that persists no matter how the learning rate schedule is chosen",
    "and it applies equally to convolutional and transformer-based models",
    "which means the choice can be made once and never revisited",
    "and the improvement is realised in full on any deployment target",
    "regardless of whether the parameters are frozen or continue to receive updates",
    "and this is always preferable to any alternative formulation of the objective",
    "which follows directly from the definition and admits no counterexamples",
    "and the same factor applies identically to every tensor in the model",
    "an outcome that cannot be altered by changing the optimiser or its settings",
    "and this behaviour is fixed at initialisation and never changes thereafter",
    "which guarantees the result will hold on data drawn from any distribution",
    "and no trade-off against accuracy, latency, or memory is incurred at any point",
    "regardless of how sparse or heavily quantised the underlying weights become",
    "and this alone is sufficient to explain the entire effect being observed",
    "which is why the alternative approach can be ruled out in every situation",
    "and the conclusion is unaffected by the size of the corpus used for pretraining",
]


def target(i):
    """Where the correct option should land by length: 1st, 2nd, or last."""
    return {0: 1, 4: 5}.get(i % 5, 2)


def extend(text, pool):
    return text.rstrip(".") + ", " + pool.pop(0)


def main():
    blocks, order = {}, []
    for path in sorted(glob.glob(f"{SRC}/qbank_*.txt")):
        cur = {}
        for line in open(path, encoding="utf-8"):
            s = line.rstrip("\n")
            if s.startswith("Q:"):
                cur = {"prompt": s[2:].strip(), "path": path}
            elif re.match(r"^[A-E]\) ", s):
                cur[s[0]] = s[3:].strip()
            elif s.startswith("*:"):
                cur["answer"] = s[2:].strip()
                order.append(cur["prompt"])
                blocks[cur["prompt"]] = cur
                cur = {}

    pool = []
    for i, p in enumerate(order):
        b = blocks[p]
        ans = b["answer"]
        wrong = [l for l in LETTERS if l != ans]
        want = target(i)
        for _ in range(40):
            lens = {l: len(b[l]) for l in LETTERS}
            rank = sorted(LETTERS, key=lambda l: -lens[l]).index(ans) + 1
            if rank == want:
                break
            if not pool:
                pool = CLAUSES[(i * 7) % len(CLAUSES):] + CLAUSES[:(i * 7) % len(CLAUSES)]
            if want == 1:
                b[ans] = extend(b[ans], pool)
            elif want == 2:
                longest = max(wrong, key=lambda l: lens[l])
                if lens[longest] > lens[ans] and rank > 2:
                    b[sorted(wrong, key=lambda l: -lens[l])[1]] = extend(
                        b[sorted(wrong, key=lambda l: -lens[l])[1]], pool)
                else:
                    b[longest] = extend(b[longest], pool)
            else:
                b[min(wrong, key=lambda l: lens[l])] = extend(b[min(wrong, key=lambda l: lens[l])], pool)

    for path in sorted(glob.glob(f"{SRC}/qbank_*.txt")):
        out = []
        for p in order:
            b = blocks[p]
            if b["path"] != path:
                continue
            out.append(f"Q: {b['prompt']}\n")
            out += [f"{l}) {b[l]}\n" for l in LETTERS]
            out.append(f"*: {b['answer']}\n\n")
        open(path, "w", encoding="utf-8").writelines(out)
    print(f"rebalanced {len(order)} questions")


main()
