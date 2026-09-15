# Dataset Sources

This repository does **not** redistribute the raw datasets used in the experiments.

Users should download the datasets from the original providers listed below and place them in the expected data locations before running the experiments. Users are responsible for complying with the licenses, terms of use, access requirements, and citation requirements of the original dataset providers.

## 1. AI4I 5W1H Knowledge-Unit Experiment

The AI4I experiment uses the AI4I 2020 Predictive Maintenance Dataset. This dataset is used for the predictive-maintenance case study and the 5W1H knowledge-unit experiment.

| Experiment                          | Dataset                                  | Source                                                                                                            |
| ----------------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| AI4I 5W1H Knowledge-Unit Experiment | AI4I 2020 Predictive Maintenance Dataset | UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset |

Formal citation:

```text
Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
        
        
        
        
        
        
```

## 2. Cross-domain Benchmark Datasets

The cross-domain benchmark uses public datasets covering regression, binary classification, and multiclass classification tasks. The local filenames listed below correspond to the filenames expected by the experiment configuration files. Some files may have been renamed locally during experiment preparation.

Raw datasets are not included in this repository. Download them from the original source pages and place them in the data location expected by the corresponding Colab scripts.

## 2.1 Regression Datasets

| No. | Experiment dataset name   | Expected local file        | Source                                                                                                                                            |
| --: | ------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
|   1 | `insurance`               | `insurance.csv`            | Medical Cost Personal Datasets, Kaggle: https://www.kaggle.com/datasets/mirichoi0218/insurance                                                    |
|   2 | `life_expectancy`         | `Life Expectancy Data.csv` | Life Expectancy (WHO), Kaggle: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who                                                   |
|   3 | `car_selling_price`       | `car data.csv`             | Vehicle Dataset from CarDekho, Kaggle: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho                                   |
|   4 | `cancer_mortality_rates`  | `cancer_reg.csv`           | Health Outcomes and Socioeconomic Factors, Kaggle: https://www.kaggle.com/datasets/thedevastator/uncovering-trends-in-health-outcomes-and-socioec |
|   5 | `flight_inflight_service` | `train.csv`                | Airline Passenger Satisfaction, Kaggle: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction                                |
|   6 | `auto_mpg`                | `auto-mpg.csv`             | Auto-mpg Dataset, Kaggle: https://www.kaggle.com/datasets/uciml/autompg-dataset                                                                   |
|   7 | `bicycles_rent`           | `day.csv`                  | Bike Sharing Dataset, Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset                                                |
|   8 | `concrete`                | `Concrete_Data.csv`        | Concrete Compressive Strength Data, Kaggle: https://www.kaggle.com/datasets/vivekgediya/concrete-data                                             |
|   9 | `loan_amount`             | `credit_risk_dataset.csv`  | Credit Risk Dataset, Kaggle: https://www.kaggle.com/datasets/laotse/credit-risk-dataset                                                           |
|  10 | `energy_heating_load`     | `ENB2012_data.csv`         | Energy Efficiency Dataset, Kaggle: https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset                                              |
|  11 | `energy_cooling_load`     | `ENB2012_data.csv`         | Energy Efficiency Dataset, Kaggle: https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset                                              |
|  12 | `forest_fire`             | `forestfires.csv`          | Forest Fires Data Set, Kaggle: https://www.kaggle.com/datasets/elikplim/forest-fires-data-set                                                     |
|  13 | `online_news_popularity`  | `OnlineNewsPopularity.csv` | UCI Online News Popularity Data Set, Kaggle: https://www.kaggle.com/datasets/thehapyone/uci-online-news-popularity-data-set                       |
|  14 | `car_price`               | `CarPrice_Assignment.csv`  | Car Price Prediction Multiple Linear Regression, Kaggle: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction                            |
|  15 | `house_values`            | `housing.csv`              | California Housing Prices, Kaggle: https://www.kaggle.com/datasets/camnugent/california-housing-prices                                            |
|  16 | `cereal_rating`           | `cereal.csv`               | 80 Cereals, Kaggle: https://www.kaggle.com/datasets/crawford/80-cereals                                                                           |
|  17 | `abalone`                 | `abalone.csv`              | Abalone Dataset, Kaggle: https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset                                                            |
|  18 | `electrical_energy`       | `Folds5x2_pp.csv`          | Combined Cycle Power Plant, Kaggle: https://www.kaggle.com/datasets/ibrahimkaratas/folds                                                          |
|  19 | `real_estate`             | `Realestate.csv`           | Real Estate Valuation by UCI, Kaggle: https://www.kaggle.com/datasets/dskagglemt/real-estate-valuation-by-uci                                     |

## 2.2 Binary Classification Datasets

| No. | Experiment dataset name | Expected local file    | Source                                                                                                                                         |
| --: | ----------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
|   1 | `hotel_bookings`        | `hotel_bookings.csv`   | Hotel Booking Demand, Kaggle: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand                                               |
|   2 | `adult_income`          | `adult.csv`            | Adult Census Income, Kaggle: https://www.kaggle.com/datasets/uciml/adult-census-income                                                         |
|   3 | `diabetes`              | `diabetesWithhead.csv` | Diabetes Dataset, Kaggle: https://www.kaggle.com/datasets/johndasilva/diabetes                               |
|   4 | `titanic`               | `titanic.csv`          | Titanic - Machine Learning from Disaster, Kaggle: https://www.kaggle.com/c/titanic/data                                                        |
|   5 | `candy`                 | `candy-data.csv`       | The Ultimate Halloween Candy Power Ranking, Kaggle: https://www.kaggle.com/datasets/fivethirtyeight/the-ultimate-halloween-candy-power-ranking |
|   6 | `churn`                 | `Churn.csv`            | Telco Customer Churn, Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn                                                   |
|   7 | `credit_card`           | `creditcard.csv`       | Credit Card Fraud Detection, Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud                                                   |
|   8 | `credit_delinquency`    | `credit.csv`           | Give Me Some Credit, Kaggle: https://www.kaggle.com/c/GiveMeSomeCredit/data                                                                    |
|   9 | `bank`                  | `bank.csv`             | Bank Marketing, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/222/bank+marketing                                        |

## 2.3 Multiclass Classification Datasets

| No. | Experiment dataset name | Expected local file       | Source                                                                                                        |
| --: | ----------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------- |
|   1 | `atomic`                | `train(1).csv`            | Superconductivty Data Data Set, Kaggle: https://www.kaggle.com/datasets/tunguz/superconductivty-data-data-set |
|   2 | `iris`                  | `Iris.csv`                | Iris Species, Kaggle: https://www.kaggle.com/datasets/uciml/iris                                              |
|   3 | `glass`                 | `glass.csv`               | Glass Classification, Kaggle: https://www.kaggle.com/datasets/uciml/glass                                     |
|   4 | `redwine`               | `classwinequalityred.csv` | Wine Quality Dataset, Kaggle: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009                   |
|   5 | `student_mat`           | `student-mat.csv`         | Student Alcohol Consumption, Kaggle: https://www.kaggle.com/datasets/uciml/student-alcohol-consumption        |
|   6 | `payment_delays`        | `cs-training.csv`         | Give Me Some Credit, Kaggle: https://www.kaggle.com/c/GiveMeSomeCredit/data                                   |

## 3. Dataset Placement

The experiment configuration files specify the expected filenames. Raw datasets should be downloaded from the source links above and placed in the data location expected by the corresponding experiment scripts.

For the AI4I 5W1H Knowledge-Unit Experiment, the recommended structure is:

```text
experiments/
└── ai4i_5w1h_knowledge_unit/
    └── data/
        └── [AI4I dataset files downloaded from UCI]
```

The cross-domain benchmark was run in Google Colab. Its scripts are provided under:

```text
experiments/cross_domain_benchmark/colab_scripts/
```

For the cross-domain benchmark, download the required datasets and make them available to the Colab runtime, for example by uploading them to the Colab session or mounting Google Drive. The dataset directory used in Colab should match the paths expected by the corresponding scripts and configuration files under:

```text
experiments/cross_domain_benchmark/input_configs/
```

If Google Drive is mounted, the corresponding data path should be updated in the Colab scripts before execution.

## 4. Non-redistribution Statement

The raw datasets are not included in this repository for the following reasons:

* the datasets are maintained by their original providers;
* some datasets have specific licensing or access conditions;
* some Kaggle competition datasets require account-level access or agreement to competition terms;
* redistributing raw datasets may violate provider-specific terms of use.

This repository therefore provides only source links, expected filenames, experiment configuration files, scripts, prompts, and summary outputs.
