# EvoProto simplified KD experiment

One predeclared arm: original no_graph seed correction (70.4831 reference), replacing KD reliability heuristics with the existing seed-supported novel-region veto.

For valid pixels predicted as old foreground by the frozen teacher, KD uses original student-confusion degree and class balancing. Novel seeds and their existing connected feature-supported extensions veto KD in both main and prototype heads. Remove teacher sigmoid confidence, old CAM threshold/support weight, new CAM suppression, and PAR-new veto from KD selection. Keep seed correction, SEP anchors/losses, prototype geometry, branch weights, temperature 2, and old-foreground conditional KL unchanged. The standard changed-pixel exclusion remains for SEP/PTC/KD; full novel regions additionally veto KD only.

Seed regions are computed from predictions/image tags and frozen teacher features, never pixel ground truth. Region identification reuses the existing seed/connected-expansion mechanism, including its pre-existing thresholds. This simplifies KD selection, not the seed extractor itself.

Protocol: GPUs 0/1 authorized, global batch8, seed0, crop448, Step1 and Step2 each8000 updates from the same protected Step0; no OLC, no ALD, no extra training. Only final weights saved. Final square448/aspect672 evaluation and fixed alpha0.5 fusion; no GT tags at segmentation inference. Compare with cached identical-protocol baseline and prior no_graph; single-seed results do not establish stable gains.

Code isolated under this directory. Original experiments and original protected checkpoints are not modified. Runner releases its own GPU reservation after successful final evaluation.
