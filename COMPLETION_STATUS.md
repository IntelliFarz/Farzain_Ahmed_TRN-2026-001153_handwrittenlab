# Completion status — read before submission

## Implemented in this package
- Seed fixed to S=1153 throughout the runner.
- Part A: sklearn train/test split, scaled KNN and logistic regression, random forest, own-sheet preprocessing, 5×10 plot, comparison plot, handwriting confusion matrix, CSV/Markdown accuracy table.
- Part B: NumPy 64–32–10 network, forward/loss/backward/update logic, finite-difference gradient checks, training-loss plot, zero-weight/no-ReLU/100×-LR experiments.
- Part C: PyTorch 64–32–10 model, DataLoader, six learning rates, validation split from training data, combined loss plot, validation table, saved/reloaded model prediction check.
- Part D: seed-dependent CNN, hand-vs-code parameter count, sklearn and handwriting evaluation, filters, feature maps, 100-sample overfitting plot, stride-2 output-size calculation.
- Part E: corrected training script and five conceptual bug descriptions in a table.
- Part F1: manual-formula NumPy scaled dot-product attention matrix and outputs for S=1153.
- Part F2: 20 original example review prompts/labels prepared; model inference is optional and requires `transformers` plus internet access.

## Must be checked/completed by the student
1. Inspect `outputs/handwriting_5x10_grid.png`. Automatic crop quality is not guaranteed by a single photograph. Fix cell/row crop coordinates in `load_handwriting()` if any digit is clipped or includes table lines. Confirm exactly 50 correctly labeled samples before reporting handwriting scores.
2. The provided sheet's rows are assumed to be five repetitions of 0,1,...,9. Verify this order.
3. Photograph handwritten derivations with roll reference 1153: Part B gradients; Part D layer sizes and parameter counts; Part F1 token-1 attention calculation.
4. For Part C, the exact NumPy-to-PyTorch copied-weight equivalence and five gradient comparisons should be added/checked if required for grading; current C section focuses on the six-rate study and save/reload.
5. Part D's two improvement ideas should be run and compared on the student's own verified handwriting set. The current CNN includes weight decay but does not establish that two ideas improved handwriting accuracy.
6. Part E's table lists the five flaws and corrected held-out evaluation. To satisfy the assignment's one-at-a-time evidence, execute each single-fix intermediate version and record its observed score; do not invent intermediate accuracies.
7. Part F2: edit/confirm your 20 labels, install transformers, run the pipeline, save model labels/scores, compute English and Roman Urdu results, inspect three wrong cases and tokenizer outputs.
8. Add at least 10 genuine dated experiment-log entries as experiments happen. Do not backdate or invent entries.
9. Complete the four-page maximum report and 10-minute presentation using the actual run outputs. Include three failed experiments and what they taught you.
10. Add all AI prompts used to the appendix. Do not write “I used no AI tools.”

## Results generated in this environment (re-run locally)
The runner was executed here with S=1153. Its output files are included in `outputs/` and `tables/`. These are preliminary results, not a substitute for rerunning in your own VS Code environment. In particular, handwriting accuracy depends on verifying the image crops.
