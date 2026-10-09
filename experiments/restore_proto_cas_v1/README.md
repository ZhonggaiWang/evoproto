# Confusion-guided class ambiguity supervision (CAS, experimental)

This experiment tests whether candidate-set supervision reduces old-to-new overwriting.
It is distinct from main/prototype inference fusion. OLC and old ALD are disabled.
Candidate-label learning already exists; this experiment does not pre-claim novelty or improvement.

## Selection

Old teacher predicts foreground a with sigmoid confidence >=0.7 and a positive old image prediction.
PAR proposes new class b with a current image tag. Main and auxiliary CAM both support b at >=0.25, but they do not both confidently select b at >=0.7. Reuse the existing .25/.7 thresholds.
The current detached prototype head must prefer a over b. Exclude crop padding.
The confusion variant additionally requires the directed OLD-to-NEW edge a->b to be a supported top-2 confusion neighbor (at least 3 distinct images).
All selection evidence is detached; use the previous iteration's graph. No modification during the first 2000 warmup updates.

The initial strong-CAM rule selected predominantly real new-class pixels in a fixed 200-image validation audit (Step1 98.54% new; Step2 100%). This motivated focusing on weak CAM propagation and requiring current prototype corroboration. The initial code and diagnostics remain preserved, and this adaptive design choice must be disclosed.

## Marginal BCE

For logits z and candidates S={a,b}, sum the probabilities of two mutually exclusive independent-Bernoulli configurations: only a positive, or only b positive.

L_set = sum_c softplus(z_c) - logsumexp(z_a,z_b).

This does not require both classes positive. Singleton S recovers original one-hot BCE exactly.
Replace selected pixels' main and prototype hard losses. Preserve original valid-pixel denominator and original loss coefficients .1/.1.
Remove the same pixels from PTC hard relations, SEP anchors and KD valid mask to prevent contradictory targets.
Image classification targets, confusion estimation, remaining pixel losses and old prototype direction preservation are unchanged.
Update the graph from original trusted anchors before masking the loss-side anchors.

## Controls and protocol

- baseline: reuse completed full EvoProto without OLC.
- confusion_pair: graph-guided candidate supervision.
- ignore: same selection rule and hard-constraint removal, no candidate loss; retain original denominator to isolate additional set supervision.
- local_pair: same local evidence and teacher-old/CAM-new pair, without graph filtering. This is a local candidate control, not student softmax top-2.

Each formal arm independently starts from shared Step0. Each Step1/2 has 8000 updates, global batch8 (4/GPU), seed0, physical 4090 GPUs5/6. Step2 uses its own Step1 final checkpoint. Save final weights only.
Train crop448; evaluate square448 and aspect-preserving area672. Report main and pre-fixed .5 prototype fusion. No best-intermediate checkpoint or fusion-weight selection.

## Verification

CPU tests cover baseline loss/gradient equivalence, exact marginal likelihood, zero ignored gradients and selection guards.
Two-GPU smoke tests cover baseline/off/ignore/local_pair/confusion_pair, both stages.
Audit uses 200 fixed evenly-spaced validation images. Pixel GT enters metric calculation only AFTER candidate selection and never enters training or graph updates.
Training cas_metrics.jsonl samples one globally aggregated batch every50 updates; it is not an interval total.
Final comparison verifies sources, lineage, budget, per-image histograms, and paired image bootstrap. One seed and adaptively inspected VOC validation cannot establish robustness or sufficient novelty.

## Run

Start reservation.py with CUDA_VISIBLE_DEVICES set to the authorized GPU5/6 UUIDs. It has a 12-hour maximum lifetime. Formal runner releases reservation on completion/failure.

CPU: `.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_cas_v1/test_supervision.py`

Smoke: `.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_cas_v1/run.py --smoke --arms baseline off ignore local_pair confusion_pair`

Formal: `.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_cas_v1/run.py --arms confusion_pair ignore local_pair`
