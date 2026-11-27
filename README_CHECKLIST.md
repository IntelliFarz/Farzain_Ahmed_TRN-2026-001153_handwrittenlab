# The Handwriting Lab: submission checklist

Name: Farzain Ahmed | Trainee ID: TRN-2026-001153 | Course: Artificial Intelligence | Trainer: Lajvanti Ratnani
Seed S = (last 4 digits of your roll number): ________   (confirm with your trainer which number counts as your roll number)

ZIP name (from the brief): name_traineeid_handwrittenlab.zip  ->  Farzain_Ahmed_TRN-2026-001153_handwrittenlab.zip
(The PDF says rollnumber_handwriting_lab.zip; the LMS brief says name_traineeid_handwrittenlab.zip. Ask your trainer if unsure.)

## What must be in the ZIP
- [ ] Notebooks for Parts A to F (folder `notebooks/`), running top to bottom with seed S
- [ ] Photos of your handwriting sheet(s) with your roll number (`handwriting_sheets/`) and the 50 processed 8x8 images (`processed_digits/`)
- [ ] Photos of hand derivations: Part B backpropagation, Part D sizes and parameters, Part F attention (`derivations/`)
- [ ] Bug table for Part E and sentence table for Part F2 (`tables/`)
- [ ] Experiment log with at least 10 dated entries, written as you work (`experiment_log.md`)
- [ ] Report, PDF, 4 pages maximum, and presentation slides (`report/`, `slides/`)
- [ ] AI usage appendix: every prompt you used, or the sentence "I used no AI tools" (`AI_USAGE_APPENDIX.md`)

## Rules to remember (they cost marks)
1. Use your seed S everywhere: np.random.seed(S), torch.manual_seed(S), random_state=S.
2. The test photos must be YOUR handwriting, with your roll number on every sheet.
3. Every number in the report must come from a notebook cell the teacher can re-run.
4. AI is for learning only. Declare every prompt. Undeclared AI use counts as copying.
5. In the presentation you must explain any line of your own code. Practise this.
6. A wrong result with a good explanation earns more than a perfect number with no explanation.

## Suggested order (work back from the LMS deadline; the PDF says 2 weeks from 15 Nov 2026)
1. Part A (collect handwriting early, the photos take time)
2. Part B (derive on paper first), then Part C
3. Part D, then Parts E and F
4. Part G: write one log entry every day you work; build the summary table; write the 4-page report; make the slides; rehearse the live demo with a new digit

## Where your photos are in this folder
- handwriting_sheets/digits_sheet_1_original.jpg : your 50 test digits (add your roll number to the sheet and re-photograph)
- notes_handwritten/ : your 8 pages of study notes (not the required derivations)
- derivations/ : still empty. Put the 3 required derivation photos here.
- processed_digits/ : still empty. Your own preprocessing code must create the 50 processed 8x8 images here.
