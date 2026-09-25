# RNN-Forecasting
Machine Learning CS441: Time Series Forecasting using Recurrent Neural Networks

# About the Dataset

A collection of popular benchmark datasets for **time series forecasting**. From **giochelavaipiatti** on kaggle
at https://www.kaggle.com/datasets/giochelavaipiatti/time-series-forecasts-popular-benchmark-datasets

## Datasets

### [Traffic](#ref-1)

Contains hourly traffic data from San Francisco freeway car lanes:

* **Long-term forecasting:** 429 car lanes
* **Collection period:** From 2015-01-01
* **Sampling interval:** 1 hour

### [Electricity](#ref-2)

Contains electricity consumption data from individual clients:

* **Short-term forecasting:** 370 clients
* **Long-term forecasting:** 321 clients
* **Collection period:** From 2011-01-01
* **Sampling interval:** 15 minutes

### [COVID-19](#ref-3)

Contains COVID-19 hospitalization data for the U.S. state of California (CA), provided by Johns Hopkins University.

* **Collection period:** 2020-01-02 to 2020-12-31
* **Sampling interval:** 1 day


### [Weather](#ref-4)

Contains 21 meteorological indicators, including humidity and air temperature, collected from the Weather Station of the Max Planck Institute for Biogeochemistry in Germany.

* **Collection period:** 2020
* **Sampling interval:** 10 minutes
* **Number of indicators:** 21

### [ETT](#ref-5)

The **Electricity Transformer Temperature (ETT)** dataset is collected from two different electric transformers, with two temporal resolutions:

* **ETTh:** 1 hour
* **ETTm:** 15 minutes

## References

<a id="ref-1"></a>

1. [Multivariate Time Series Data — GitHub](https://github.com/laiguokun/multivariate-time-series-data)

<a id="ref-2"></a>

2. [ElectricityLoadDiagrams20112014 — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)

<a id="ref-3"></a>

3. [Weather Station — Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/)


<a id="ref-4"></a>

4. [Weather Station — Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/)

<a id="ref-5"></a>

5. [ETDataset — GitHub](https://github.com/zhouhaoyi/ETDataset)

6. Yi, K., Zhang, Q., Fan, W., Wang, S., Wang, P., He, H., An, N., Lian, D., Cao, L., & Niu, Z. (2023). *Frequency-domain MLPs are More Effective Learners in Time Series Forecasting*. Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS 2023). [OpenReview](https://openreview.net/forum?id=iif9mGCTfy)
