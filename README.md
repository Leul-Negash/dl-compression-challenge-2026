# Advanced DL Compression Challenge 2026

Distilling DeBERTa-v3-large into a 90 MB student that answers deep-learning
multiple-choice questions at 10 ms per question on a T4.

| Metric | Result | Limit |
| --- | --- | --- |
| MAP@3 (public leaderboard) | 0.93750 | |
| MAP@3 (5-fold CV, held out) | 0.9421 | |
| Model size on disk | 90.26 MB | 150 MB |
| Average latency | 10.4 ms / question | 30 ms |
| Size multiplier | 1.000 | |
| Latency multiplier | 1.000 | |

`Final Grade = MAP@3 x min(1, 30/latency_ms) x min(1, 150/size_MB)`, so both
multipliers sit at their cap and the grade is the MAP@3 untouched.

## Problem

Transformers hit a memory wall in deployment. Attention is quadratic in sequence
length, and moving weight matrices through GPU memory dominates latency during
inference. On edge hardware an 8B model is simply not runnable.

The task: take a large teacher model, compress it below 500M parameters through
distillation, pruning and quantization, and answer a hidden set of A-E multiple
choice questions on advanced deep learning topics. Scored on MAP@3, then
multiplied down if the artifact exceeds 150 MB or 30 ms per question.

72 training questions are provided. 108 test questions are scored, split into a
32-row public leaderboard and a 76-row private one.

## Pipeline

**Teacher.** `DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, 435M parameters.
Starting from an NLI checkpoint rather than the raw pretrained model matters a
lot here: entailment is close to multiple-choice scoring, and with only 72
training examples there is not enough signal to learn that relationship from
scratch. Trained 4 epochs per fold across 5 folds, batch size 2 with gradient
checkpointing to fit on a T4.

**Distillation.** Each fold teacher predicts on every row. The student for fold
*f* is trained against the logits from teacher *f* only. Loss is
`0.7 * KL(student/T || teacher/T) * T^2 + 0.3 * CE(student, label)` with `T = 3`.

**Student.** `DeBERTa-v3-base-mnli-fever-anli`, 88.8M parameters after the
vocabulary trim below.

**Pruning.** Global unstructured L1 pruning at 25% over all `nn.Linear` weights,
followed by 5 recovery epochs, then `prune.remove` to fold the masks into the
weights. Measured sparsity is 25.0% across linear layers and 24.1% across the
whole model.

**Quantization.** Post-training INT8, symmetric, one scale per output row, applied
to every 2-D floating point tensor. Everything else stays FP16.

**Extra: vocabulary trimming.** See finding 2.

**Extra: written training data.** See finding 5.

## Key findings

### 1. Quantizing for latency and quantizing for size are different problems

The obvious route from the course notes is `torch.quantization.quantize_dynamic`.
Its kernels are CPU-only. Latency here is scored as total inference time divided
by 108 rows, so the entire run has to finish inside 3.24 seconds to stay under
30 ms per question. On the 2 vCPUs of a free Kaggle instance that misses by
roughly an order of magnitude.

What the grader actually measures is `os.path.getsize` of the saved artifact. So
INT8 is used for **storage**, and the weights are dequantized to FP16 and run on
the T4 at inference. The file on disk is INT8; the forward pass is not. This is
the single decision that keeps both multipliers at 1.000.

### 2. The embedding table was more than half the model

DeBERTa-v3 ships a 128,000-token SentencePiece vocabulary. At 768 dimensions that
is 98M parameters of embedding against roughly 86M of transformer, so the
embedding table alone was larger than the network using it. Quantized to INT8 the
full model came to about 184 MB, over the 150 MB cap.

Both `train.csv` and `test.csv` are available up front and internet is off, so the
set of token IDs the model can ever see is fixed and knowable. Tokenizing every
prompt and option in both files yields 3,652 distinct IDs. Keeping only those
rows of the embedding matrix and remapping input IDs through a lookup array takes
the artifact from 184 MB to 90.26 MB with no risk of an out-of-vocabulary token,
because the vocabulary is derived from the exact text the model will be asked
about.

### 3. Unstructured pruning bought no size reduction at all

Pruning is a required phase, and the accuracy cost is small, but it is worth
being clear about what it does and does not do here. The artifact stores dense
INT8 tensors, so a zeroed weight still occupies one byte. Running the pipeline at
30% and at 25% sparsity produced files of identical size, 90,261,468 bytes both
times.

Every megabyte saved came from quantization and the vocabulary trim. Unstructured
sparsity only pays off with a sparse storage format or hardware that exploits the
pattern, neither of which applies to a `torch.save` of dense tensors on a T4.

The practical consequence: since 30% and 25% cost the same on disk, there is no
reason to prune harder than the requirement. The pipeline uses 25%, which clears
the 20% floor whether sparsity is measured over the pruned linear layers (25.0%)
or the whole network including embeddings and biases (24.1%).

### 4. Cross-validation leaked through the distillation targets

The first working version scored 0.9769 on held-out CV, with a pruned 90 MB
student apparently beating its own 435M teacher by six points. That was a bug in
my setup, not a result.

Teacher predictions were pooled into a single out-of-fold array, and the student
for fold *f* was distilled against that array's training rows. But each of those
out-of-fold logits came from a teacher that had trained on fold *f*'s validation
rows. Information about the held-out questions reached the student through the
teacher's parameters, even though no held-out row was ever in the student's
training set.

The fix is to keep each fold teacher's predictions separately and give the
student for fold *f* only the targets from teacher *f*, which never saw that
fold's validation rows. Corrected CV came out at 0.9375, four points lower. That
number then matched the public leaderboard exactly on first submission, which is
what gave me confidence the harness was honest.

Worth stating plainly: the inflated number looked like a great result and was
entirely an artifact of how the targets were built.

### 5. More training data moved the teacher but not the student

72 examples is the real constraint, so I wrote 116 additional A-E questions
covering the same four areas as the test set (transfer learning and PEFT,
self-supervised learning, model compression, LLM systems).

Two things had to be handled carefully:

- **Contamination.** My first draft included questions whose prompts restated
  provided test questions almost verbatim. A synthetic question that repeats a
  test prompt with an answer attached is hand-labelling the test set by another
  route, which the rules prohibit. Every question above 0.45 token Jaccard against
  any provided prompt was dropped, 65 in total. Maximum overlap in what remains is
  0.444, which is just shared topic vocabulary.
- **Option length.** In the provided data the correct option is the second-longest
  of the five 58% of the time, well above the 20% chance rate, because the long
  option is usually an over-qualified wrong one. Written from scratch my correct
  answers were the longest 176 times out of 181, which teaches the opposite cue.
  `tools/balance_aug.py` extends selected distractors with absolutist qualifiers
  until the length profile matches the source: 21/59/20 percent for
  longest/second-longest/shortest against the provided data's 18/58/18.

CV folds are built from the 72 provided questions only. The written questions
join every fold's training set and never its validation set, so the resulting
number stays comparable.

Result: teacher CV went from 0.9167 to 0.9421, a gain of about 1.8 questions. The
student went from 0.9375 to 0.9421, which is one third of one question and inside
the noise. The extra data helped the 435M teacher and did not transfer to the
88.8M student.

### 6. Neither scoreboard can resolve differences this small

MAP@3 credit comes in thirds, so on the 32-row public leaderboard the smallest
non-zero difference between two submissions is `(1/3)/32 = 0.0104`. Every gap I
dealt with was exactly that size: the difference between my two best models, and
my margin over second place, were both one tick.

The same model scored 0.9421 on CV (one tick better than the baseline) and
0.92708 on the public leaderboard (one tick worse). Both measurements are correct
and they disagree, because 32 and 72 questions are not enough to separate models
this close. After this became clear I stopped tuning against the public score,
since at 32 rows one lucky guess is worth three points of nothing.

## Repository layout

```
notebooks/01_train.py     training pipeline, source form
notebooks/01_train.ipynb  same, split into cells for Kaggle
notebooks/02_infer.py     inference, internet off
notebooks/02_infer.ipynb  same, split into cells for Kaggle
tools/to_ipynb.py         splits a .py on "# %%" markers into a notebook
tools/build_aug.py        compiles the question banks into augment.csv
tools/balance_aug.py      matches option-length profile to the provided data
data/aug/qbank_*.txt      the 116 written questions, source form
data/aug/augment.csv      compiled, uploaded to Kaggle as a dataset
```

The competition CSVs are not in this repository. They are distributed through the
competition page and are not mine to redistribute.

## Reproducing

`01_train` runs on Kaggle with the competition attached plus the `augment.csv`
dataset, GPU on, internet on for the checkpoint downloads. It takes about 72
minutes on a T4 and writes `student_int8.pt`, the tokenizer and the model config.

`02_infer` attaches that output, runs with **internet off** as the rules require,
and writes `submission.csv` along with the two lines the grading protocol
expects:

```
Model Size: 90.26
Average Latency: 0.010404 seconds per sample
```

Latency is measured after a three-batch GPU warmup, over the full 108 rows,
padded to the true maximum length of 115 tokens rather than the 256 used in
training.
