# Optional ALD refinement candidate — not yet GPU-validated or launched

This directory prepares an optional final-stage adaptation of ALD V9 for the explicit-prototype restoration. It is not the current reported final method. The complete `restore_proto_v1` chains and their ablations continue independently. Activate this candidate only after corrected square448/aspect672 and fusion evaluations are available, and after GPU smoke checks.

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

The entry `train.py` requires `--step 2 --spg 4 --no-pretrained --loss_warmup_iters 0 --start-checkpoint <own-stage2-final> --prev_checkpoint <own-stage1-final> --initial-confusion <own-stage2-confusion.json>`; enable `--refine-ald --image-evidence <own-cache.json>` only for ALD arms. Supply the existing VOC paths and an isolated work directory. GPU cache generation, both-arm smoke checks, candidate activation, formal training and results remain pending. No benefit is claimed yet.

Both GPU entry points require `CUDA_VISIBLE_DEVICES=GPU-82e069d7-189e-06b0-faed-b08e1794a981,GPU-2d0a204c-101b-1149-06e1-e3528d9f38bf` and reject an unset/different mapping. Do not launch them while the formal restoration coordinator is still using these cards.
