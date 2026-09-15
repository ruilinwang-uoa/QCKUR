# CWRU Bearing Knowledge-Unit Experiment

This directory contains the CWRU bearing-vibration experiment used as the second predictive-maintenance instantiation of QCKUR. The experiment evaluates six reporting systems on a curated 40-record subset of the CWRU 12 kHz drive-end corpus and includes a separate 16-record outer-race position robustness probe.

The experimental design is described in `docs/CWRU_EXPERIMENT_PROTOCOL.md`.

## Repository Contents

The CWRU experiment includes:

```text
cwru_bearing_knowledge_unit/
├── README.md
├── requirements.txt
├── data/
│   └── bearing_cwru/
│       └── cwru_manifest.json
├── docs/
│   └── CWRU_EXPERIMENT_PROTOCOL.md
├── knowledge/
│   ├── ku_inventory_bearing.json
│   ├── maps/
│   │   └── physics_rules_spec_bearing.json
│   └── question_bank/
│       └── question_bank_bearing.json
├── src/
├── runs/
└── analysis/
    └── bearing_main_summary.csv
```

Raw CWRU `.mat` files are not stored in the repository. `src/download_bearing_cwru.py` downloads the 40 records used by the main benchmark. `src/bearing_probe_positions.py` obtains and evaluates the separate outer-race position-probe files when that robustness check is run.

## Environment

Install the experiment dependencies:

```bash
pip install -r requirements.txt
```

LLM-dependent stages read credentials and model settings from environment variables. The generation stage uses the provider key `openai` in the scripts for compatibility with the OpenAI-style client interface; by default this maps to the Zhipu OpenAI-compatible endpoint and `glm-4-flash`. The Layer-2/Layer-3 judging stage uses the `deepseek` provider key and defaults to `deepseek-v4-flash`.

The relevant environment variables are:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_MODEL

DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL
DEEPSEEK_MODEL
```

A local `.env` file at the experiment root is also supported by `src/llm_config.py`.

## Reproduction

The main workflow is:

```bash
python src/download_bearing_cwru.py
python src/bearing_v3.py
python src/bearing_probe_positions.py
python src/bearing_knowledge.py --emit-bank --gt
python src/bearing_campaign.py --run B1 B2 B3 B4 B5 B8
python src/bearing_l1_eval.py
python src/bearing_l2l3_eval.py --run
python src/bearing_l2l3_eval.py --summary
```

`bearing_v3.py` constructs the frozen two-tier diagnostic handler. `bearing_knowledge.py` creates the bearing-specific knowledge units and programmatic reporting ground truth. `bearing_campaign.py` generates the six report sets. The Layer-1 and Layer-2/3 scripts evaluate the saved reports under the three-layer protocol.

## Saved Artifacts

The repository currently includes the following CWRU-specific artifacts:

* `data/bearing_cwru/cwru_manifest.json` — benchmark manifest and condition labels.
* `knowledge/ku_inventory_bearing.json` — bearing-specific knowledge-unit inventory.
* `knowledge/maps/physics_rules_spec_bearing.json` — bearing diagnostic-rule specification.
* `knowledge/question_bank/question_bank_bearing.json` — bearing reporting question bank.
* `runs/bearing_gt.json` — programmatic reporting ground truth.
* `runs/bearing_reports_B1.parquet` through `runs/bearing_reports_B5.parquet`, and `runs/bearing_reports_B8.parquet` — saved reports.
* `runs/bearing_l1.csv` — Layer-1 evaluation output.
* `runs/bearing_l2.parquet` and `runs/bearing_l3.parquet` — Layer-2 and Layer-3 evaluation outputs.
* `analysis/bearing_main_summary.csv` — aggregate result table used for the CWRU comparison.

The 40-record main benchmark and the 16-record position robustness probe are kept separate so that probe records do not enter the primary evaluation set.
