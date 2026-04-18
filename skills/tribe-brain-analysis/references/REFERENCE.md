# TRIBE v2 Model Reference

## Architecture

TRIBE v2 is a multi-modal brain encoding model that predicts group-averaged
cortical BOLD responses on the fsaverage5 surface (20,484 vertices, 2 Hz).

### Encoders

| Encoder | Input | Purpose |
|---------|-------|---------|
| V-JEPA2 | Video frames (480p, 24 fps) | Visual feature extraction |
| wav2vec-BERT | Audio (16 kHz mono) | Auditory feature extraction |
| Llama-3.2-3B | Text / transcript | Semantic feature extraction |

### Decoder

A transformer decoder maps the concatenated multi-modal features to the
fsaverage5 cortical surface at 2 Hz (500 ms TR).

## Training data

- 25 subjects from the NSD (Natural Scenes Dataset) and video fMRI studies
- Group-averaged to produce a single population-level predictor
- Not suitable for individual subject predictions

## Limitations

1. **Group average only** — not individual fMRI
2. **50 s maximum** — hard cap from training (`duration_trs=100` at 2 Hz)
3. **5 s hemodynamic lag** — pre-applied during training
4. **CC-BY-NC 4.0 license** — no commercial use of model weights

## Atlas reference

### Yeo-7 networks

| Code | Full name |
|------|-----------|
| Vis | Visual |
| SomMot | Somatomotor |
| DorsAttn | Dorsal Attention |
| SalVentAttn | Salience / Ventral Attention |
| Limbic | Limbic |
| Cont | Frontoparietal Control |
| Default | Default Mode |

### Schaefer-400 labels

Labels follow the format: `7Networks_{hemi}_{network}_{index}`
- `hemi`: LH (left hemisphere) or RH (right hemisphere)
- `network`: Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default
- `index`: ROI number within the network

### Harvard-Oxford labels

Prefix `HO-cort:` for cortical, `HO-sub:` for subcortical.
Subcortical regions include: Thalamus, Caudate, Putamen, Pallidum,
Hippocampus, Amygdala, Accumbens.

### Jülich cytoarchitectonic labels

Prefix `Juelich:`. Maps to Brodmann areas:
- BA17 = V1 (primary visual cortex)
- BA44/45 = Broca's area (language production)
- BA22 = Wernicke's area (language comprehension)
- BA4 = Primary motor cortex
- BA1/2/3 = Primary somatosensory cortex

## Temporal dynamics metrics

| Metric | Definition |
|--------|------------|
| `peak_s` | Time of max mean |z| across all vertices |
| `rise_s` | First TR where global |z| crosses 50% of peak |
| `duration_above_half_max_s` | TRs above 50% of peak |
| `decay_slope_per_tr` | Linear regression slope from peak to end |

## Citation

If using TRIBE v2 in published work, cite the original paper and model card.
Model weights are licensed CC-BY-NC 4.0 (non-commercial use only).
