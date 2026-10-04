# Dataset Sources

This repository does **not** redistribute the raw datasets used in the experiments.

Users should obtain the datasets from the original providers listed below and place them in the locations expected by the corresponding experiment scripts. Users are responsible for complying with the licenses, terms of use, access requirements, and citation requirements of the original dataset providers.

Source links and dataset metadata in this document were checked on **2026-09-30**.

## 1. AI4I 5W1H Knowledge-Unit Experiment

The AI4I experiment uses the **AI4I 2020 Predictive Maintenance Dataset** for the predictive-maintenance case study and the 5W1H knowledge-unit experiment.

| Experiment | Dataset | Primary source |
| --- | --- | --- |
| AI4I 5W1H Knowledge-Unit Experiment | AI4I 2020 Predictive Maintenance Dataset | UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset |

Formal citation:

```text
Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.
UCI Machine Learning Repository.
https://doi.org/10.24432/C5HS5C
```

Recommended local location:

```text
experiments/
└── ai4i_5w1h_knowledge_unit/
    └── data/
        └── [AI4I dataset files downloaded from UCI]
```

## 2. CWRU Bearing Knowledge-Unit Experiment

The CWRU experiment uses vibration recordings from the **Case Western Reserve University Bearing Data Center**. The official site provides normal-bearing data and seeded single-point bearing faults recorded under different motor loads. CWRU states that the data files are MATLAB (`.mat`) files containing drive-end (`DE`) and fan-end (`FE`) vibration signals, motor speed (`RPM`), and, for some recordings, base (`BA`) vibration signals.

Primary official sources:

- Bearing Data Center / data access: https://engineering.case.edu/bearingdatacenter/download-data-file
- Apparatus and procedures: https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures
- Normal baseline data: https://engineering.case.edu/bearingdatacenter/normal-baseline-data
- 12 kHz drive-end fault data: https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
- Bearing information: https://engineering.case.edu/bearingdatacenter/bearing-information

A convenience mirror of the CWRU **Normal Baseline Data** and **12 kHz Drive End Bearing Fault Data** is available on Zenodo:

```text
Case Western Reserve University Bearing Dataset Pt.1
Zenodo record: https://zenodo.org/records/10986655
DOI: https://doi.org/10.5281/zenodo.10986655
```

The official CWRU source should be treated as the primary provenance source. The Zenodo mirror is used only as a reproducibility/download fallback and is not the original producer of the data.

### 2.1 Main 40-record benchmark

The main CWRU experiment uses **40 recordings**: four normal baselines plus 36 drive-end fault recordings. The fault recordings are from the official **12 kHz Drive End Bearing Fault Data** collection and cover three fault locations (inner race, ball, outer race), three defect diameters (0.007, 0.014, and 0.021 inches), and four motor loads (0, 1, 2, and 3 hp). The main outer-race condition uses the 6 o'clock position.

| Condition | Defect diameter | CWRU file IDs (loads 0, 1, 2, 3 hp) |
| --- | ---: | --- |
| Normal | — | `97.mat`, `98.mat`, `99.mat`, `100.mat` |
| Inner race | 0.007 in | `105.mat`, `106.mat`, `107.mat`, `108.mat` |
| Inner race | 0.014 in | `169.mat`, `170.mat`, `171.mat`, `172.mat` |
| Inner race | 0.021 in | `209.mat`, `210.mat`, `211.mat`, `212.mat` |
| Ball | 0.007 in | `118.mat`, `119.mat`, `120.mat`, `121.mat` |
| Ball | 0.014 in | `185.mat`, `186.mat`, `187.mat`, `188.mat` |
| Ball | 0.021 in | `222.mat`, `223.mat`, `224.mat`, `225.mat` |
| Outer race @ 6 o'clock | 0.007 in | `130.mat`, `131.mat`, `132.mat`, `133.mat` |
| Outer race @ 6 o'clock | 0.014 in | `197.mat`, `198.mat`, `199.mat`, `200.mat` |
| Outer race @ 6 o'clock | 0.021 in | `234.mat`, `235.mat`, `236.mat`, `237.mat` |

The repository manifest is:

```text
experiments/cwru_bearing_knowledge_unit/data/bearing_cwru/cwru_manifest.json
```

The downloader is:

```text
experiments/cwru_bearing_knowledge_unit/src/download_bearing_cwru.py
```

It downloads the required `.mat` files from the official CWRU site when available and uses the Zenodo mirror as a fallback. Raw `.mat` files are intentionally not committed to this repository.

### 2.2 Outer-race position robustness probe

A separate 16-record probe evaluates generalization to outer-race positions not used in the main benchmark. These records are kept separate from the primary 40-record evaluation set.

| Condition | Defect diameter | Position | CWRU file IDs (loads 0, 1, 2, 3 hp) |
| --- | ---: | --- | --- |
| Outer race | 0.007 in | 3 o'clock | `144.mat`, `145.mat`, `146.mat`, `147.mat` |
| Outer race | 0.007 in | 12 o'clock | `156.mat`, `158.mat`, `159.mat`, `160.mat` |
| Outer race | 0.021 in | 3 o'clock | `246.mat`, `247.mat`, `248.mat`, `249.mat` |
| Outer race | 0.021 in | 12 o'clock | `258.mat`, `259.mat`, `260.mat`, `261.mat` |

The probe is implemented in:

```text
experiments/cwru_bearing_knowledge_unit/src/bearing_probe_positions.py
```

Recommended local location for both the main benchmark and probe files:

```text
experiments/
└── cwru_bearing_knowledge_unit/
    └── data/
        └── bearing_cwru/
            ├── cwru_manifest.json
            ├── 97.mat
            ├── 98.mat
            └── ...
```

## 3. MetroPT In-Service Fleet Experiment

The MetroPT experiment uses three released telemetry streams from the Air Production Unit (APU) of metro vehicles operated in Porto, Portugal. The streams include pressure, temperature, motor-current, digital control/status signals, and, for the 2022 releases, GPS/flow information. The source datasets are real operational time series intended for predictive maintenance, anomaly detection, failure prediction, and related analyses.

The repository uses the internal tags `metropt3`, `metropt2022`, and `metropt2022B` to distinguish the three evaluated vehicle-epochs. These tags are **repository identifiers**, not official dataset names.

### 3.1 MetroPT-3 (`metropt3`, 2020 vehicle-epoch)

Primary source:

```text
MetroPT-3 Dataset
UCI Machine Learning Repository, dataset ID 791
https://archive.ics.uci.edu/dataset/791/metropt+3+dataset
DOI: https://doi.org/10.24432/C5VW3R
```

UCI distributes:

```text
MetroPT3(AirCompressor).csv
Data Description_Metro.pdf
```

The UCI record describes the dataset as a multivariate time series collected from an in-service train APU. It contains 1,516,948 observations and 15 sensor features. The released CSV spans 2020 and the UCI record includes company-reported failure periods. UCI lists the dataset under **CC BY 4.0**.

Formal dataset citation supplied by UCI:

```text
Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021).
MetroPT-3 Dataset [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5VW3R
```

Repository downloader:

```text
experiments/metropt_5w1h/src/download_metropt3.py
```

Expected local files:

```text
experiments/metropt_5w1h/data/metropt3/MetroPT3(AirCompressor).csv
experiments/metropt_5w1h/data/metropt3/Data Description_Metro.pdf
```

The repository does **not** redistribute these raw source files. The PDF is source documentation supplied by UCI and should be obtained directly from UCI rather than committed to this repository.

### 3.2 MetroPT 2022 stream (`metropt2022`, repository "train A")

The 2022 MetroPT dataset was released on Zenodo and described in the Scientific Data data descriptor below.

Original release (v1):

```text
MetroPT: A Benchmark dataset for predictive maintenance
Zenodo record: https://zenodo.org/records/6854240
DOI: https://doi.org/10.5281/zenodo.6854240
Source file: dataset_train.csv
```

The later **V2** record also contains this same `dataset_train.csv` file and is the recommended single source when reproducing all 2022 streams:

```text
MetroPT2: A Benchmark dataset for predictive maintenance
Zenodo record: https://zenodo.org/records/7766691
DOI: https://doi.org/10.5281/zenodo.7766691
```

In this repository, the downloaded `dataset_train.csv` file is renamed locally to:

```text
dataset_train_2022.csv
```

This rename is only to make its role unambiguous. The word `train` in the original deposited filename is the depositor's filename and should **not** be interpreted as a machine-learning train/test split.

Expected local file:

```text
experiments/metropt_5w1h/data/metropt3/dataset_train_2022.csv
```

Repository provenance manifest:

```text
experiments/metropt_5w1h/data-manifests/dataset_train_2022_manifest.json
```

### 3.3 MetroPT2 stream (`metropt2022B`, repository "train B")

The second 2022 stream is distributed in the V2 Zenodo record:

```text
MetroPT2: A Benchmark dataset for predictive maintenance
Zenodo record: https://zenodo.org/records/7766691
DOI: https://doi.org/10.5281/zenodo.7766691
Source file: MetroPT2.csv
```

The Zenodo record reports 7,116,940 instances, 21 attributes, nominal 1 Hz logging, and a time span from 2022-04-28 to 2022-07-28. It also provides two company-reported failure periods (one air leak and one oil leak).

Expected local file:

```text
experiments/metropt_5w1h/data/metropt3/MetroPT2.csv
```

Repository provenance manifests:

```text
experiments/metropt_5w1h/data-manifests/metropt2_manifest.json
experiments/metropt_5w1h/data-manifests/metropt2_trainB_manifest.json
```

### 3.4 MetroPT data descriptor and label provenance

The principal data descriptor for the 2022 MetroPT release is:

```text
Veloso, B., Ribeiro, R. P., Gama, J., & Pereira, P. M. (2022).
The MetroPT dataset for predictive maintenance.
Scientific Data, 9, 764.
https://doi.org/10.1038/s41597-022-01877-3
```

The data descriptor reports an operating-train APU stream collected in 2022 at 1 Hz, with ground-truth anomaly/failure information derived from the operator's maintenance reports.

The raw MetroPT datasets are not fully labelled sample-by-sample. Therefore, the event windows, reconciled failure metadata, exclusions, and repository-specific vehicle-epoch identifiers used in the manuscript are documented separately in the repository manifests and adjudication materials. Those repository annotations should not be confused with the raw providers' original files.

Generic Zenodo downloader:

```text
experiments/metropt_5w1h/src/download_zenodo_file.py
```

Example commands from the repository root (adjust paths if your checkout layout differs):

```bash
python experiments/metropt_5w1h/src/download_metropt3.py

python experiments/metropt_5w1h/src/download_zenodo_file.py \
  7766691 dataset_train.csv \
  experiments/metropt_5w1h/data/metropt3/dataset_train_2022.csv

python experiments/metropt_5w1h/src/download_zenodo_file.py \
  7766691 MetroPT2.csv \
  experiments/metropt_5w1h/data/metropt3/MetroPT2.csv
```

## 4. Cross-Domain Benchmark Datasets

The cross-domain benchmark uses public datasets covering regression, binary classification, and multiclass classification tasks. The local filenames listed below correspond to the filenames expected by the experiment configuration files. Some files may have been renamed locally during experiment preparation.

Raw datasets are not included in this repository. Download them from the original source pages and place them in the data location expected by the corresponding Colab scripts.

### 4.1 Regression Datasets

| No. | Experiment dataset name | Expected local file | Source |
| --: | --- | --- | --- |
| 1 | `insurance` | `insurance.csv` | Medical Cost Personal Datasets, Kaggle: https://www.kaggle.com/datasets/mirichoi0218/insurance |
| 2 | `life_expectancy` | `Life Expectancy Data.csv` | Life Expectancy (WHO), Kaggle: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who |
| 3 | `car_selling_price` | `car data.csv` | Vehicle Dataset from CarDekho, Kaggle: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho |
| 4 | `cancer_mortality_rates` | `cancer_reg.csv` | Health Outcomes and Socioeconomic Factors, Kaggle: https://www.kaggle.com/datasets/thedevastator/uncovering-trends-in-health-outcomes-and-socioec |
| 5 | `flight_inflight_service` | `train.csv` | Airline Passenger Satisfaction, Kaggle: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction |
| 6 | `auto_mpg` | `auto-mpg.csv` | Auto-mpg Dataset, Kaggle: https://www.kaggle.com/datasets/uciml/autompg-dataset |
| 7 | `bicycles_rent` | `day.csv` | Bike Sharing Dataset, Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset |
| 8 | `concrete` | `Concrete_Data.csv` | Concrete Compressive Strength Data, Kaggle: https://www.kaggle.com/datasets/vivekgediya/concrete-data |
| 9 | `loan_amount` | `credit_risk_dataset.csv` | Credit Risk Dataset, Kaggle: https://www.kaggle.com/datasets/laotse/credit-risk-dataset |
| 10 | `energy_heating_load` | `ENB2012_data.csv` | Energy Efficiency Dataset, Kaggle: https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset |
| 11 | `energy_cooling_load` | `ENB2012_data.csv` | Energy Efficiency Dataset, Kaggle: https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset |
| 12 | `forest_fire` | `forestfires.csv` | Forest Fires Data Set, Kaggle: https://www.kaggle.com/datasets/elikplim/forest-fires-data-set |
| 13 | `online_news_popularity` | `OnlineNewsPopularity.csv` | UCI Online News Popularity Data Set, Kaggle: https://www.kaggle.com/datasets/thehapyone/uci-online-news-popularity-data-set |
| 14 | `car_price` | `CarPrice_Assignment.csv` | Car Price Prediction Multiple Linear Regression, Kaggle: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction |
| 15 | `house_values` | `housing.csv` | California Housing Prices, Kaggle: https://www.kaggle.com/datasets/camnugent/california-housing-prices |
| 16 | `cereal_rating` | `cereal.csv` | 80 Cereals, Kaggle: https://www.kaggle.com/datasets/crawford/80-cereals |
| 17 | `abalone` | `abalone.csv` | Abalone Dataset, Kaggle: https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset |
| 18 | `electrical_energy` | `Folds5x2_pp.csv` | Combined Cycle Power Plant, Kaggle: https://www.kaggle.com/datasets/ibrahimkaratas/folds |
| 19 | `real_estate` | `Realestate.csv` | Real Estate Valuation by UCI, Kaggle: https://www.kaggle.com/datasets/dskagglemt/real-estate-valuation-by-uci |

### 4.2 Binary Classification Datasets

| No. | Experiment dataset name | Expected local file | Source |
| --: | --- | --- | --- |
| 1 | `hotel_bookings` | `hotel_bookings.csv` | Hotel Booking Demand, Kaggle: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand |
| 2 | `adult_income` | `adult.csv` | Adult Census Income, Kaggle: https://www.kaggle.com/datasets/uciml/adult-census-income |
| 3 | `diabetes` | `diabetesWithhead.csv` | Diabetes Dataset, Kaggle: https://www.kaggle.com/datasets/johndasilva/diabetes |
| 4 | `titanic` | `titanic.csv` | Titanic - Machine Learning from Disaster, Kaggle: https://www.kaggle.com/c/titanic/data |
| 5 | `candy` | `candy-data.csv` | The Ultimate Halloween Candy Power Ranking, Kaggle: https://www.kaggle.com/datasets/fivethirtyeight/the-ultimate-halloween-candy-power-ranking |
| 6 | `churn` | `Churn.csv` | Telco Customer Churn, Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| 7 | `credit_card` | `creditcard.csv` | Credit Card Fraud Detection, Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| 8 | `credit_delinquency` | `credit.csv` | Give Me Some Credit, Kaggle: https://www.kaggle.com/c/GiveMeSomeCredit/data |
| 9 | `bank` | `bank.csv` | Bank Marketing, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/222/bank+marketing |

### 4.3 Multiclass Classification Datasets

| No. | Experiment dataset name | Expected local file | Source |
| --: | --- | --- | --- |
| 1 | `atomic` | `train(1).csv` | Superconductivty Data Data Set, Kaggle: https://www.kaggle.com/datasets/tunguz/superconductivty-data-data-set |
| 2 | `iris` | `Iris.csv` | Iris Species, Kaggle: https://www.kaggle.com/datasets/uciml/iris |
| 3 | `glass` | `glass.csv` | Glass Classification, Kaggle: https://www.kaggle.com/datasets/uciml/glass |
| 4 | `redwine` | `classwinequalityred.csv` | Wine Quality Dataset, Kaggle: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009 |
| 5 | `student_mat` | `student-mat.csv` | Student Alcohol Consumption, Kaggle: https://www.kaggle.com/datasets/uciml/student-alcohol-consumption |
| 6 | `payment_delays` | `cs-training.csv` | Give Me Some Credit, Kaggle: https://www.kaggle.com/c/GiveMeSomeCredit/data |

## 5. Dataset Placement

The experiment configuration files and download scripts specify the expected filenames and locations. A high-level layout is:

```text
experiments/
├── ai4i_5w1h_knowledge_unit/
│   └── data/
│       └── [AI4I raw data]
├── cwru_bearing_knowledge_unit/
│   └── data/
│       └── bearing_cwru/
│           └── [CWRU .mat files]
├── metropt_5w1h/
│   └── data/
│       └── metropt3/
│           ├── MetroPT3(AirCompressor).csv
│           ├── Data Description_Metro.pdf
│           ├── dataset_train_2022.csv
│           └── MetroPT2.csv
└── cross_domain_benchmark/
    ├── colab_scripts/
    └── input_configs/
```

For the cross-domain benchmark, the experiments were run in Google Colab. Download the required datasets from the source links above and make them available to the Colab runtime, for example by uploading them to the session or mounting Google Drive. The dataset directory should match the paths expected by the corresponding configuration files and notebooks.

## 6. Non-Redistribution and Provenance Statement

Raw datasets are intentionally excluded from this repository because:

- the datasets are maintained by their original providers;
- licenses and redistribution terms differ across providers;
- some Kaggle datasets or competitions require account-level access or acceptance of provider-specific terms;
- the official CWRU Bearing Data Center does not provide a repository-level redistribution license in the pages used here, so the repository links to the source rather than republishing the recordings;
- keeping raw third-party data outside the repository avoids confusing source data with the derived manifests, annotations, windows, and evaluation outputs produced by this project.

The repository therefore provides source links, expected filenames, download scripts where available, manifests, experiment configurations, prompts, derived features, annotations, and summary outputs. Provider-supplied documentation files should be downloaded from the original provider rather than copied into the public repository unless their redistribution terms clearly permit it.
