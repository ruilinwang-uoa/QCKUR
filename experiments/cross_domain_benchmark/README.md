# Cross-domain Benchmark Experiment

This folder contains the Google Colab-based materials used for the cross-domain benchmark reported in the paper.

The benchmark evaluates the QCKUR prototype against LLM-based baselines across regression, binary classification, and multiclass classification tasks. It is designed to assess whether question-centered, template-guided answer generation improves the factual reliability of analytical answers across different datasets, task types, and LLM backends.

The benchmark contains **34 public datasets** and **306 unique question instances**:

* 9 binary-classification datasets: 81 question instances
* 6 multiclass-classification datasets: 54 question instances
* 19 regression datasets: 171 question instances

Each question was generated **five times per method–LLM condition**, yielding **1,530 generated answers per condition**.

## Contents

The main contents are:

```text
colab_scripts/
input_configs/
docs/
```

The `colab_scripts/` folder contains the Google Colab notebooks used to run the QCKUR prototype and LangChain single-agent experiments.

The `input_configs/` folder contains JSON configuration files specifying the datasets, expected filenames, task types, model settings, background knowledge, independent variables, dependent variables, and user questions used in the benchmark.

The `docs/GuidanceAndTasks.md` file contains the human-evaluation instructions, correctness criteria, A–E error taxonomy, and evaluation checklists used for the benchmark.

The prototype notebooks also use the `qckur_project/` folder, which contains the prototype components required by the Colab scripts:

```text
qckur_project/
├── datasciencecomponents.py
├── NLGcomponents.py
└── apptemplates/
```

Raw datasets are not redistributed in this repository. Dataset sources and expected filenames are documented in:

```text
../../doc/DATASETS.md
```

## Running in Colab

The benchmark was run in Google Colab. The notebooks are intended to reproduce the experimental workflow in a Colab environment rather than to serve as standalone local command-line scripts.

### QCKUR Prototype

For the prototype notebooks, prepare the following Colab-side layout:

```text
/content/
└── qckur_project/
    ├── datasciencecomponents.py
    ├── NLGcomponents.py
    ├── apptemplates/
    ├── configs/
    │   └── [experiment config JSON files]
    └── data/
        └── [dataset CSV files]
```

Before running a prototype notebook:

1. Upload or copy the required JSON configuration files from `input_configs/` in this repository to:

```text
/content/qckur_project/configs/
```

2. Upload the corresponding dataset files to:

```text
/content/qckur_project/data/
```

3. Ensure that the filenames match those specified in the selected JSON configuration file.

Typical paths used by the prototype notebooks are:

```python
PROJECT_DIR = "/content/qckur_project"
CONFIG_PATH = "/content/qckur_project/configs/[config_file].json"
DATA_DIR = "/content/qckur_project/data"
```

### LangChain Single-Agent Baseline

For the single-agent baseline notebooks, the expected Colab-side layout is simpler:

```text
/content/
├── [experiment config JSON files]
└── [dataset CSV files]
```

Typical paths are:

```python
CONFIG_PATH = "/content/[config_file].json"
DATA_DIR = "/content"
```

If Google Drive is used instead of direct upload to the Colab runtime, update `PROJECT_DIR`, `CONFIG_PATH`, and `DATA_DIR` in the corresponding notebook before running the cells.

## Repeated Generations

Each question instance was generated **five times under each evaluated method–LLM condition**.

Some notebooks produce one generation for each question during a single execution. In those cases, the notebook was executed repeatedly to obtain the five generations required by the experimental protocol. Where a notebook provides an internal repeated-run parameter, the corresponding setting can be used instead.

The five generations for each question are treated as repeated observations of the same experimental condition rather than as additional unique question instances.

## API Keys

API keys are not included in this repository.

Users should provide their own API credentials in Colab, preferably through Colab Secrets or environment variables. Credentials should not be written directly into notebooks or committed to the repository.

The exact model endpoint and model version may need to be updated if the original API version is no longer available.

## Human Evaluation

The cross-domain benchmark was evaluated under a human-judged protocol.

Thirty evaluators with mathematics or statistics backgrounds assessed anonymized generated answers under blinded conditions. Answers were first judged as correct or incorrect. Incorrect answers were then classified using the following five-category taxonomy:

* **A:** hallucination or unsupported content
* **B:** incomplete answer or missing necessary caveat
* **C:** numeric-only response or insufficient explanation
* **D:** incorrect interpretation
* **E:** other error

The participant/evaluator instructions and corresponding checklists are provided in:

```text
docs/GuidanceAndTasks.md
```

## Outputs

The notebooks save generated outputs to the Colab runtime.

Prototype outputs are typically saved under paths such as:

```text
/content/prototype_[model_setting]_outputs_[llm]/
```

Single-agent baseline outputs are typically saved under paths such as:

```text
/content/langchain_generic_outputs_[llm]/
```

Generated outputs may include:

* per-dataset generated-answer files
* combined output JSON files
* runtime records
* token-usage records
* failure logs

Because Colab runtime storage is temporary, generated outputs should be saved to Google Drive or downloaded after each experimental run.

Direct LLM generation was conducted separately and therefore does not have a corresponding executable baseline notebook in this repository.

## Reproducibility Notes

The JSON configuration files preserve the experimental inputs used in the cross-domain benchmark, while the Colab notebooks preserve the execution procedure for the QCKUR prototype and LangChain single-agent baseline.

The benchmark uses **34 datasets, 306 unique question instances, and five repeated generations per method–LLM condition**. These quantities should be preserved when reproducing or extending the experiments.

Exact generated answers may vary because of changes in model versions, API providers, endpoint implementations, decoding behavior, and Colab runtime conditions. Consequently, reproduction should focus on preserving the experimental inputs, model settings, question sets, generation protocol, and evaluation procedure rather than expecting byte-identical generated text.

Raw datasets and API keys are not redistributed in this repository.
