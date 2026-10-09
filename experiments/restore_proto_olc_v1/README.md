# EvoProto + OLC (Old-class Label Correction)

OLC replaces the rejected extra-training ALD candidate. Each incremental stage still receives exactly 8,000 optimizer updates, batch size 8 (4 on each of physical GPUs 5 and 6), seed 0, and the original learning-rate schedule. Step 1 starts from the shared step-0 final checkpoint; step 2 starts from its own step-1 final checkpoint. There is no additional refinement stage, same-stage reference network, partial-set loss, or reference KL loss.

## Method

The frozen previous-stage teacher already produces main and auxiliary image-classification logits. For each image ID and each head, OLC maintains an exponential moving average of sigmoid probabilities over the randomly augmented training views observed so far. On its first visit, memory equals the current prediction. Subsequent visits use `memory = 0.5 * memory + 0.5 * current`. Duplicate IDs within one global batch are averaged before updating. Both ranks maintain identical detached memory. The fixed momentum 0.5 is an initial design choice, not a measured optimum.

An old class is positive when both averaged head probabilities exceed 0.5, negative when neither exceeds 0.5, and unknown otherwise. The teacher weights remain frozen; it is the per-image prediction memory that evolves inside the original 8,000-step training.

- Old-class classification BCE uses positive and negative decisions. Unknown entries contribute zero gradient, with the original batch-times-class denominator preserved. Current-class ground-truth image labels are unchanged.
- Unknown classes remain in CAM competition so that their spatial evidence is not automatically relabeled as background. Their winning old-class CAM/PAR labels are then ignored for hard supervision and PTC.
- Teacher old-class pixel labels are retained only for OLC-positive image classes. Rejected old-class labels become ignore (255), never background. Valid current-class pseudo labels retain priority.
- The original confusion/prototype mechanism is unchanged. Its existing class-tag inputs now receive corrected positives; hence unreliable old-class evidence cannot create anchors or reliable KD support.

No old-class annotations or incremental pixel masks are inputs to OLC. Offline label audits join image annotations only after predictions have been produced. Whole-image diagnostic quality does not establish the quality of EMA predictions under random crops; online memory will be audited separately.

## Evaluation

Primary comparison: main-head mIoU, same square-448 and aspect-preserving-672 protocols as the completed no-OLC baseline. Secondary comparison: the previously fixed 0.5 main/prototype probability fusion; do not retune its coefficient for OLC. Report old/new class performance as well as overall mIoU. Label diagnosis reports precision, recall counting abstained positives as misses, coverage, and wrong negatives.

At 2,000/4,000/6,000/8,000 updates, record segmentation, prototype, main CAM and auxiliary CAM mIoU. CAM diagnostics use validation image-level tags, as in the historical CAM evaluation; segmentation inference uses no image tags. These protocols must be identified separately.

Only final model weights are saved per stage. The small prediction-memory file is overwritten at evaluation boundaries. Necessary predecessor/final method weights are protected.

## Reproduce

Run from the project root with the configured runtime:

```text
.runtime/restore_proto/venv/bin/python -B -m torch.distributed.run --master_port=49374 --nproc_per_node=2 tools/test_olc.py --ddp
.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_olc_v1/run.py --smoke --arms baseline control olc
.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_olc_v1/run.py --arms olc
```

The runner records source hashes, predecessor hashes and commands, uses the existing GPU reservation guard, and restores reservation after each job or failure. It refuses to overwrite partial runs. The `control` arm uses the new trainer with OLC disabled; the `baseline` arm uses the original trainer for smoke equivalence checking. Formal comparison reuses the completed, identical-budget original full chain only after this equivalence check.

Status: two-stage GPU smoke and two-rank memory tests passed; the formal OLC chain is running. The disabled-OLC smoke differs by 0.00214 mIoU points from baseline; an unchanged-baseline repeat differs by 0.00172 points. Runs are not bitwise deterministic, and no exact-equality claim is made. No OLC segmentation gain is claimed yet. Old ALD extra-refinement results are excluded from this method.


## Step 1 completed; full-chain conclusion still pending

Step 1 finished all 8,000 updates. Its final square448 main-head mIoU is **75.28924**, compared with **74.60657** without OLC (**+0.68267 points**). The prototype head is **74.76473** versus **73.89129** (+0.87344); old foreground improves by 0.42290 points and new foreground by 1.29887 points. CAM / auxiliary CAM diagnostics are 72.16626 / 74.65226, using validation image tags; segmentation inference uses no tags.

| Step-1 updates | No-OLC main | OLC main | Change |
| --- | ---: | ---: | ---: |
| 2,000 | 47.08372 | 45.93328 | -1.15043 |
| 4,000 | 71.51848 | 73.73630 | +2.21782 |
| 6,000 | 73.97367 | 74.57842 | +0.60475 |
| 8,000 | 74.60657 | 75.28924 | +0.68267 |

Do not substitute the peak intermediate difference for the final stage difference. Independent evaluation confirms square448 main 75.28933 (less than 0.001 points from training validation), and aspect672 main 76.16801. These are 16-class Step-1 results, not the final 21-class benchmark.

Final Step-1 checkpoint SHA256: `43dfc9d5e614d141137f7c5a6e5043797924e22c79c0b559f1cedf7903e5c9db`. Finite-value/prototype-block checks passed. Step 2 is running from this exact checkpoint, with two verified workers on physical GPUs 5 and 6 and batch 4 per rank.

The final Step-1 memory audit gives precision 60.2092%, recall 90.5618%, and coverage 98.5005%. Main-head-only decisions from the same EMA give precision 54.2200% and recall 93.5955%. Screening removes 512 false positives while excluding 81 true positives relative to that same-memory comparison. These annotations are joined offline and never enter training. Two correlated teacher heads can still agree on a wrong class, and crop predictions are not equivalent to whole-image presence labels.

Final evaluation will run the no-OLC reference and OLC sequentially on the same GPU, using identical square448/aspect672 inputs and the already fixed 0.5 prototype fusion. This changes evaluation hardware from the earlier CPU reference, not the training budget or prediction rule. Both endpoints will be checked against independent evaluation before comparison.
