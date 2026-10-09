# Historical extra-training ALD candidate — excluded from the current method

The two full-model arms completed GPU validation and 1,200 extra training updates. Main-head aspect672 mIoU was 70.09234 for matched continuation and 70.56434 with the old ALD candidate (square448: 69.01756 and 69.07516). These are extra-training results, not the same-budget EvoProto baseline. On 2026-10-09 the user rejected this refinement protocol. Do not launch further arms from this directory as part of the current study. The new OLC module in `experiments/restore_proto_olc_v1` corrects old-class image labels inside the original 8,000 updates per stage. Historical design details follow for provenance only.

## Controlled protocol

- Each arm starts from its **own** 8000-update stage2 model, uses its own stage1 teacher, and freezes a copy of its own stage2 model as the reference. Never reuse a historical 71.x model or another arm's reference.
- Physical GPUs 5 and 6 only, two ranks, batch 4 per rank, global batch 8. Train 1200 additional updates (9600 image exposures), fresh AdamW, encoder LR 2e-6 / head LR 2e-5, cosine decay, no warmup. The old V9 used 300 updates at batch32 with inherited optimizer moments; this is an adaptation, not an exact reproduction.
- Compare full+ALD against full+equal-budget continuation without ALD. If ALD becomes the final method, apply it to without-confusion and without-prototype using each arm's own lineage.
- Preserve ordinary/prototype supervision and the existing confusion-guided prototype KD/SEP/direction terms. ALD changes target reliability; it does not remove prototypes. The without-prototype arm continues to disable all incremental prototype losses.
- Keep only each phase's final checkpoint. Smoke checkpoints are disposable after their logs and checks are retained.

## Evidence and supervision

`cache.py` regenerates a compact JSON of old-class states for the 2145 current-stage training images. It reads training images and **current-new image tags only**. Frozen old main/aux classifier votes, old/reference segmentation predictions on aligned original+flip views, and reference multiscale CAMs decide present/absent/unknown. Positive presence requires three native pixels of stable reference/CAM support. Unknown is never treated as an absent class. Segmentation annotations and old GT image tags are not consumed.

`ald.py` reuses V9's hard-target policy: admitted old class plus teacher/reference agreement, new PAR/reference agreement, and teacher/reference background agreement. Conflicts remain unknown; padding is ignored. Unknowns receive an allowed-class partial-set constraint plus foreground-conditional reference KL on the ordinary head. The conditional KL does not impose foreground mass or a background-logit gradient. Prototype supervision uses the calibrated hard labels and retains its existing KD/SEP terms.

Old confusion anchors require admitted presence. New confusion anchors remain CAM/PAR based, without a new reference-agreement gate. PTC discards new reference conflicts. Old KD receives the trusted-old mask. These reliability changes interact with the relation losses and must be reported, not described as perfectly independent modules.

`continued_graph.py` preserves the parent graph's EMA, mass and certified >=3-image edges. New support counts are tracked separately; old and new counts are not summed because their images may overlap. This avoids inventing historical image IDs, which were not saved in the compact parent graph.

## Checks and remaining work

CPU semantic checks cover unknown-versus-negative labels, conflicts, padding, teacher stop-gradient, conditional-FG invariance, two-view evidence and graph continuation. A two-rank CPU check compares ALD loss/gradients against the concatenated batch. Run:

```sh
.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_ald_v1/test_semantics.py
.runtime/restore_proto/venv/bin/python -B -m torch.distributed.run --master_addr=127.0.0.1 --master_port=49369 --nproc_per_node=2 -m experiments.restore_proto_ald_v1.test_distributed
```

The entry `train.py` requires `--step 2 --spg 4 --no-pretrained --loss_warmup_iters 0 --start-checkpoint <own-stage2-final> --prev_checkpoint <own-stage1-final> --initial-confusion <own-stage2-confusion.json>`; enable `--refine-ald --image-evidence <own-cache.json>` only for ALD arms. Supply the existing VOC paths and an isolated work directory. GPU cache generation, both-arm smoke checks and the full-model matched pair are complete. Further old-ALD ablations are canceled under the revised scope.

Both GPU entry points require `CUDA_VISIBLE_DEVICES=GPU-82e069d7-189e-06b0-faed-b08e1794a981,GPU-2d0a204c-101b-1149-06e1-e3528d9f38bf` and reject an unset/different mapping. Do not launch them while the formal restoration coordinator is still using these cards.

## Queue entry

`run.py` defaults to `--mode plan` and launches nothing. It verifies the original formal coordinator has exited, its state is completed, and only the recorded reservation process occupies physical cards 5/6. It restores full reservation between cache, training and evaluation jobs and on failure. Do not bypass this check.

After activation is justified, run `--mode smoke --arms full_control full_ald`; then `--mode formal --arms full_control full_ald --decision '<record the actual comparison evidence>'`. Formal admission requires the corrected full-model evaluation, verified fusion diagnostics, matching smoke source hashes and model lineage, and successful two-rank gradient/frozen-teacher checks. It runs 1200 updates and evaluates both square448 and aspect672 with the fixed int64 evaluator. The default output attempt is `runs/restore_proto_ald_v1/v1`; source changes require a new `--attempt` and repeated smoke. The source and predecessor hashes, commands, metrics and final checkpoint hash are saved for each arm.

If ALD is retained after the matched pair comparison, smoke and formally run `--arms without_confusion_ald without_proto_ald` under the same attempt and protocol. Keep main-head results as the common primary comparison, and apply any selected inference fusion policy consistently across prototype-enabled arms. A prototype-disabled arm uses main only by definition; never blend its untrained prototype predictions to manufacture a worse control. Main-only results remain the common primary ablation.
