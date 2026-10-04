# CWRU Experiment Protocol

## Scope

This experiment is the second predictive-maintenance instantiation of the QCKUR framework. It uses the Case Western Reserve University (CWRU) 12 kHz drive-end bearing vibration corpus to evaluate whether the same knowledge-controlled reporting design transfers from tabular machine data to measured vibration signals.

## Dataset

The main benchmark contains 40 records:

- 4 normal records;
- 36 single-point fault records covering inner-race, ball, and outer-race faults;
- defect diameters of 0.007, 0.014, and 0.021 inches;
- motor loads of 0, 1, 2, and 3 hp;
- outer-race records in the main benchmark use the 6 o'clock position.

The manifest in `data/bearing_cwru/cwru_manifest.json` is the source of condition labels. Loads 0--1 are used for rule calibration and model training, while loads 2--3 are held out for evaluation.

A separate robustness probe uses 16 outer-race records from the 3 o'clock and 12 o'clock positions. These records are not part of the 40-record main benchmark.

## Two-tier diagnostic handler

The diagnostic handler contains two tiers.

1. **Rule tier.** Hilbert-envelope spectra are scanned over resonance bands. The inner-race and outer-race rules use BPFI and BPFO harmonic prominence, respectively. The calibrated thresholds are 79.5 for BPFI and 99.7 for BPFO, with a dominance requirement of 2.4 for the inner-race rule.
2. **Model-handler tier.** Records not resolved by the rule tier are classified by a class-balanced random forest with 500 trees and random seed 42. The model uses 16 deterministic file-level time- and frequency-domain features and is trained only on the load-0/1 records.

The corresponding rule specification is stored in `knowledge/maps/physics_rules_spec_bearing.json`.

## Reporting systems

Six reporting systems are evaluated on the same 40 records:

- **B1:** template-only output;
- **B2:** traditional rule-based data-to-text reporting;
- **B3:** direct LLM generation;
- **B4:** few-shot LLM generation using three examples drawn from records outside the 40-record main benchmark;
- **B5:** tool-using LLM agent;
- **B8:** proposed QCKUR framework.

The LLM-based systems use GLM-4-Flash for report generation. Layers 2--3 are judged with DeepSeek-V4-Flash using a 3,000-token completion budget. The experiment uses temperature 0.3 and random seed 42.

## Evaluation

The same three-layer evaluation structure used in the manuscript is applied:

- **Layer 1:** programmatic checking of numeric consistency, rule consistency, and fault classification;
- **Layer 2:** claim consistency, contradiction, and 5W1H coverage;
- **Layer 3:** faithfulness, completeness, 5W1H coverage, coherence, and actionability.

The composite score is

`Final = 0.35 * (100 * L1) + 0.30 * (100 * L2) + 0.35 * (20 * L3)`.

The main experiment outputs are stored under `runs/`, and the aggregated results used for the manuscript are stored under `analysis/`.
