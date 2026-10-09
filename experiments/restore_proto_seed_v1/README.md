# Seed-supported teacher correction (experimental, not yet adopted)

## Problem

A previous-stage teacher has no novel-class output. Similar novel objects can therefore be assigned to old foreground classes, rather than background. Incomplete novel CAMs leave these regions under erroneous hard old-class supervision. The goal is to identify and correct this old-class absorption of novel regions, while preserving real old objects.

## Method

The existing learnable 512-D prototypes, student-confusion graph, SEP, KD and old-prototype direction loss remain active. OLC and legacy ALD are off.

1. Select foreground anchors where current main and auxiliary CAM agree at >=.7, permitted image tags and PAR agree. Old anchors additionally require teacher agreement. Select background references where teacher/PAR agree on background and both CAM maxima are <.25.
2. On reliable novel anchors, measure a directed teacher-confusion relation: row = novel seed class; column = previous-stage teacher's prediction. EMA decay .99, synchronized across GPUs, at least 3 distinct images per edge; retain top2 supported old-foreground columns. Background contributes to row mass but is never a correction target. This is distinct from the unchanged student-confusion graph used by SEP/KD.
3. For each image, pool normalized frozen teacher encoder features at each reference class's anchors (minimum 2 feature-grid pixels). These temporary 768-D seed prototypes are not learned parameters and do not replace the original learnable prototypes.
4. A novel region can expand only through spatially connected feature-grid pixels which are at least as similar as the 10th percentile of its seed similarities, more similar to that novel prototype than other available reference prototypes, and supported by both CAMs at >=.25. At least one competing reference is required. No new trainable network or retained image/feature bank is introduced.
5. Correct only pixels currently assigned to old foreground, in these expanded regions, whose teacher old class matches the directed confusion relation. Never change reliable old anchors, current novel labels, background labels, padding or existing ignore pixels. Restore image-resolution pair gating after interpolation.
6. Replace their old hard labels with the supported novel class for both main/prototype segmentation BCE. Remove contradictory PTC/SEP/KD constraints on the same pixels. Original student-confusion estimation and every other loss remain unchanged. Activate after the original 2000-update warmup; update teacher-confusion evidence throughout training.

The graph uses the current synchronized batch's detached seed evidence. It never uses validation pixels or pixel GT. New image tags are the only current-stage training annotations; old tags remain teacher predictions.

## Predeclared comparisons

- baseline: existing completed full restored EvoProto, no OLC.
- correct: seed-supported correction gated by teacher confusion.
- ignore: identical evidence/gating but removes those old labels; preserve original BCE denominator to avoid rescaling other pixels.
- no_graph: identical prototype/seed/spatial evidence but no teacher-confusion pair gate.

All new arms start from the same fixed Step0, have their own Step1->Step2 chain, each stage8000 updates, seed0, crop448, global batch8=4/GPU, physical4090 GPUs5/6. No continuation, best-iteration selection or extra training. Final weights only; stage1 retained while needed as teacher.

Evaluate main and fixed .5 main/prototype fusion under square448 and aspect-preserving area672. Test-time segmentation uses neither image tags nor this correction module.

## Diagnostics

CPU and two-rank tests cover direction, unique-image support, graph synchronization, connected expansion, preservation of reliable old anchors, no-seed/no-reference cases, padding and input immutability.

Initial offline diagnosis calibrates the teacher graph on 200 deterministic training images through the image-only dataset, then measures corrections on 200 disjoint validation images. On final trained arms, the saved online training graph is used instead. Pixel masks enter metric calculation only after proposals are fixed. Record per-image results to expose concentration in a few objects.

At2000/4000/6000/8000 updates, 80 fixed validation images diagnose current seed precision and old-on-new vs new-on-old pseudo-label errors without updating the graph or training. These are prospective correction diagnostics; main segmentation validation remains tag-free. No intermediate model weights are saved.

The initial baseline audit found Step1 correction precision97.52%, with93576 novel pixels rescued from old labels and956 true-old pixels relabeled novel; ungated correction precision96.03%, with96150 rescues and1824 old-to-new errors. This is a limited offline diagnosis, not a final segmentation improvement. Step2 corrections were sparse (2353 valid pixels in200 images). These hypotheses still require the complete training comparison.

## Reproduction

`CUDA_VISIBLE_DEVICES='' GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 .runtime/restore_proto/venv/bin/python -B -m torch.distributed.run --master_addr=127.0.0.1 --master_port=49376 --nproc_per_node=2 experiments/restore_proto_seed_v1/test_mechanism.py --ddp`

Start the authorized GPU5/6 reservation helper, then:

`.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_seed_v1/run.py --smoke --arms baseline off correct ignore no_graph`

`.runtime/restore_proto/venv/bin/python -B experiments/restore_proto_seed_v1/run.py --arms correct ignore no_graph`

The reservation remains held between jobs and on failures until the active goal ends. Release explicitly at goal completion. Full source hashes, checkpoint lineage, commands, metrics, graph state and cleanup receipts are saved under runs/restore_proto_seed_v1.

## Limits

One seed; adaptively inspected VOC validation. Seed quality may be worse early in training than in the initial final-checkpoint audit. Teacher features may fail to separate visually similar classes. Connected expansion and relative prototypes limit but do not eliminate mistakes. The method's novelty and final segmentation benefit are not established by this pilot.
