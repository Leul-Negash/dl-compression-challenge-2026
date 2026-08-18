# Advanced DL Compression Challenge 2026

My solution for the compression challenge: a DeBERTa-v3-large teacher distilled
down to a 90 MB student that answers the multiple-choice questions in about 10
ms each on a T4.

## Results

| | Result | Limit |
| --- | --- | --- |
| Accuracy (MAP@3, public leaderboard) | 0.93750 | |
| Accuracy (MAP@3, 5-fold CV) | 0.9421 | |
| Model size | 90.26 MB | 150 MB |
| Latency | 10.4 ms per question | 30 ms |

The grade is `MAP@3 x min(1, 30/latency) x min(1, 150/size)`. Both multipliers
come out at 1.000, so nothing is lost to the size or speed penalty.

## The problem

Transformers are expensive to deploy. Attention cost grows with the square of
sequence length, and during inference most of the time goes on moving weights
through GPU memory rather than on arithmetic. On a phone or a small edge box you
cannot run an 8B model at all.

So the task is to keep as much of a large model's ability as possible while
making the file small and the inference fast. Compress a teacher below 500M
parameters using distillation, pruning and quantization, then answer a hidden
set of A-E questions about deep learning topics.

There are 72 training questions and 108 test questions. Part of the test set
feeds the public leaderboard during the competition and the rest is held back
to decide the final standing. Kaggle does not say which rows are which.

## Approach

I used `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (435M) as the teacher.
Scoring a question against five candidate answers is close to an entailment
problem, and with only 72 training examples there is very little room to learn
that mapping from scratch, so an NLI checkpoint seemed like the best use of the
data available. I did not benchmark it against the plain pretrained model, so
this is reasoning rather than a measured comparison.

Five folds, 4 epochs each, batch size 2 with gradient checkpointing so it fits
on a T4. Each fold teacher then predicts on every row.

The student is `DeBERTa-v3-base-mnli-fever-anli`, 88.8M parameters after the
vocabulary trim described below. For fold f the student is trained only against
the logits of teacher f, which is the part I got wrong the first time (see the
cross-validation section). Loss is

```
0.7 * KL(student/T || teacher/T) * T^2  +  0.3 * CE(student, label)
```

with T = 3.

After that: 25% global unstructured L1 pruning over the linear layers, 5 epochs
of recovery training, then INT8 post-training quantization with one scale per
output row. Everything that is not a 2-D float tensor stays in FP16.

## What I found

### quantize_dynamic is CPU only

The obvious move from the course notes is `torch.quantization.quantize_dynamic`.
Its kernels are CPU only, which rules it out here.

That matters because latency is scored as total inference time over 108 rows. To
stay under 30 ms per question the whole run has to finish in 3.24 seconds. Two
vCPUs running a base-sized transformer over 108 questions with five options each
is not going to do that.

The size check is separate though. It is just `os.path.getsize` on the saved
file, so there is nothing stopping you from storing INT8 and running FP16. The
weights are quantized when saved, dequantized at load time, and the forward pass
never touches an integer kernel. File is 90 MB, each question takes 10 ms.
Following the notes would have halved my grade.

### Trimming the vocabulary

DeBERTa-v3 has a 128,000 token vocabulary. At 768 dimensions that is 98M
parameters of embeddings against roughly 86M for the actual transformer. The
lookup table was larger than the network reading from it. At one INT8 byte per
parameter that is about 184 MB, over the cap. I never built that version, the
figure is just the parameter count.

Both CSVs are given up front and the inference notebook runs with no internet,
so the set of token ids the model can ever encounter is fixed and I can just
compute it. Tokenizing every prompt and option in train, test and my written
questions gives 3,652 distinct ids out of 128,000. Keeping those rows and
remapping input ids through a small lookup array drops the file to 90.26 MB. No
risk of an unknown token, since the vocabulary is built from the exact text the
model will see.

Quantization is the bigger saving in absolute terms, roughly 550 MB against the
trim's 94 MB, since it takes every weight from four bytes down to one. But INT8
on its own still left the model over the limit. The trim is what cleared the
cap.

### Pruning did not shrink anything

Pruning is required and the accuracy cost is small, but it did not shrink the
file at all.

The artifact holds dense INT8 tensors. A pruned weight is still a zero taking up
one byte. I ran the pipeline at 30% and then at 25% sparsity and the output
files were the same size to the byte, 90,261,468 both times.

Unstructured sparsity only pays off if you store it in a sparse format or run it
on hardware that skips the zeros, and neither applies to `torch.save` of dense
tensors on a T4. All the size reduction came from quantization and the
vocabulary trim.

Since 30% and 25% cost the same on disk, there was no reason to prune harder
than required. I settled on 25%, which clears the 20% floor whether you measure
over the pruned linear layers (25.0%) or over the whole network including
embeddings and biases (24.1%). Pruning at 20% would have put the whole-network
figure at 19.3%, under the line.

### A leak in my cross-validation

My first working version scored 0.9769 on held-out CV. A 90 MB pruned student
was apparently beating its own 435M teacher by six points, which is not a thing
that happens, so I went looking for the reason before trusting it.

The cause: I was pooling all the teacher predictions into one out-of-fold array
and distilling every student against it. But the out-of-fold logit for a given
row came from a teacher that had trained on other folds, including the rows I
was about to validate on. So information about the held-out questions reached
the student through the teacher's weights, even though no held-out row was ever
in the student's training data.

Fix was to keep each fold teacher's predictions separately and give student f
only the targets from teacher f, which never saw fold f's validation rows. The
corrected score was 0.9375, four points lower. That number then matched the
public leaderboard exactly on my first submission, which is what convinced me
the harness was finally honest.

Before blaming the setup I checked for duplicate questions first: one near
duplicate pair inside train, none between train and test, no reused option
strings. It was the pipeline, not the data.

### Writing extra training data

72 examples is the real bottleneck, so I wrote 116 more A-E questions covering
the same four areas as the test set: transfer learning and PEFT, self-supervised
learning, model compression, and LLM systems.

Contamination was the thing I had to watch. My first draft had questions whose
prompts restated provided test questions almost word for word. That is hand-
labelling the test set through the back door, which the rules prohibit, so I
dropped everything scoring above 0.45 token Jaccard against any provided prompt.
That removed 65 questions. The highest overlap left is 0.444, which is just
topic vocabulary that any two questions on LoRA would share.

Option length was the other one. In the provided data the correct answer is the
second-longest of the five 58% of the time, against 20% by chance, because the
longest option is usually an over-qualified wrong one. Writing from scratch I
came out the other way round: the correct answer was the longest in 176 of 181
questions. Training on that would have taught the model a cue that is
backwards for this test set. `tools/balance_aug.py` extends chosen distractors
with absolutist qualifiers until the profile matches, 21/59/20 percent for
longest/second-longest/shortest against the provided data's 18/58/18.

CV folds are built from the 72 provided questions only. The written ones go into
every fold's training set and never into validation, so the number stays
comparable.

Teacher CV went from 0.9167 to 0.9421, worth about 1.8 questions. The student
went from 0.9375 to 0.9421, which is one third of one question and inside the
noise, so most of that gain did not survive distillation.

### Measurement noise

MAP@3 gives 1, 1/2 or 1/3 credit per question, so on a leaderboard this small
the possible scores sit on a coarse grid. Judging by the values, the public
split looks like about 32 rows, which puts one question moving by a third of a
point at 0.0104.

Nearly every difference I dealt with was exactly that size, including the gap
between my two best models and my margin over second place. At one point the
same model scored a tick better than the baseline on CV and a tick worse on the
public leaderboard. Both measurements were correct.

Once that was clear I stopped tuning against the public score. With 32 questions
one lucky guess moves you three points, and chasing it is how you talk yourself
out of a model that was fine.

## Repository layout

```
notebooks/01_train.py     training pipeline
notebooks/01_train.ipynb  same thing, split into cells for Kaggle
notebooks/02_infer.py     inference, runs with internet off
notebooks/02_infer.ipynb  same, for Kaggle
tools/to_ipynb.py         splits a .py on "# %%" into a notebook
tools/build_aug.py        compiles the question banks into augment.csv
tools/balance_aug.py      matches option lengths to the provided data
data/aug/qbank_*.txt      the 116 written questions
data/aug/augment.csv      compiled, uploaded to Kaggle as a dataset
```

The competition CSVs are not here. They come from the competition page and are
not mine to hand out.

## Running it

`01_train` needs the competition data and the `augment.csv` dataset attached,
GPU on, internet on for the checkpoint downloads. About 72 minutes on a T4. It
writes `student_int8.pt` plus the tokenizer and model config.

`02_infer` takes that output, runs with internet off as required, and writes
`submission.csv` and the two grading lines:

```
Model Size: 90.26
Average Latency: 0.010404 seconds per sample
```

Latency is timed after a three batch warmup so the CUDA context and kernel
autotuning are not counted, across all 108 rows, padded to the real maximum
length of 115 tokens instead of the 256 used in training.