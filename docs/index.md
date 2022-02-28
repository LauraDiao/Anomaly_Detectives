# Network Signal Anomaly Detection
#### by the Anomaly Detectives
Laura Diao, Jenna Yang, Benjamin Sam

# Overview
In order to detect issues in network transmission data, we built a real-time-capable anomaly detection system. This system includes both alerting and monitoring features, which would enable Internet Service Providers (ISP's) such as Viasat to properly monitor user network performance and quality in real time. Moreover, detection in real-time would allow Viasat to handle issues more promptly and increase customer satisfaction. The system utilizes simulated network traffic data to train a model that predicts the packet loss rate as well as the latency of an internet connection, and uses these as well as certain features derived from the data to determine whether an adverse event is occurring.

# Quick Links
- [Main GitHub Repository](https://github.com/LauraDiao/Q2)
- [Modified DANE](https://github.com/jenna-my/modified_dane)

# Table of Contents
- [Introduction](#-Introduction)
- [Methods](#-Methods)
- [Results](#-Results)
- [Limitations](#-Limitations)
- [Conclusions](#-Conclusion)
- [Special Thanks](#-Special-Thanks)
<!-- TODO fix this TOC to make it work -->

# Introduction
Network degradation occurs in many forms, and our project will focus on two common factors: packet loss and latency. Packet loss occurs when one or more data packets transmitted across a computer network fail to reach their destination. Latency can be defined as a measure of delay for data to transmit across a network. For internet users, high rates of packet loss and significant latency can manifest in jitter or lag, indicators of overall poor network performance as perceived by the end user. Thus, when issues arise in these two factors, it would be beneficial for service providers to know exactly when the user is experiencing problems in real time. In real world scenarios, situations or environments such as poor port quality, overloaded ports, network congestion and more can impact overall network performance.

[Here's a more in depth discussion on networks and common problems with packet loss and latency](network-domain.md)

# Data
- [short summary paragraph detailing high level overview of data generation process, model building work and its performance, and why we used it (importance of dataset)]

## Data Generation with DANE
- explain who what why how dane in context of project


- explain how data is obtained (refer to repo for in depth, maybe #TODO a separate page on reproducing results)
[More in depth tutorial on how to use our modified fork of DANE](dane-details.md)

We built our model using simulated metwork data used by a network emulation tool built by a previous capstone project to predict the packet loss rate and latency of a connection. This

## Exploring the Feature Space
- A bit more eda

# Methods

## Pipeline
- short summary paragraph discussing what is and how the general pipeline works
- how pipeline was set up and basic motivations, why we wanted it and what results we wanted
- pipeline visuals, why we chose it

## The Regression Model
- the regression problem presented
- [all features used by final regression model, with their general explanations]
- how we came up with features, why/how we selected them

Our project runs on the following data pipeline as seen above. Raw data from our modified fork of DANE (either for train or test purposes) is cleaned and transformed into 10 second aggregations which are then fed into a regression model that makes predictions of loss and latency. With predictions being made on a stream of data in real time, our anomaly detector would ingest these predictions and identify anomalous behavior based on established thresholds of change.  

## Anomaly Classifier
- follow up the regression model and how this is an extension of the prev reg model
- our general mechanism (for now): threshold on pct change of regression model output

Finally, we plan to implement and integrate the anomaly detection mechanism for the system, which would ingest the predictions from our regression model and identify anomalies in 10 second windows. We fleshed out the definition and threshold of an anomaly, and defined it as any significantly rapid increase in packet loss rate as well as latency.

## Results (header/subheader)
- Regression Model performance metrics
- Classification performance metrics (gotta be figured out)
- analysis of results and power of model

[link to more in depth (and possibly interactive!) tutorial on model pipeline](model-details.md)

# Conclusions
- post modelling analysis and practical use, why this work is important

## [Possible Practical Extensions?]

## Limitations
- limitations of using simulated data as provided by DANE

Throughout our project, we ran into some limitations. Due to the size of the dataset we processed and trained on, this led to long run times and made our workflow less efficient. we combated this by reducing overcorrelated features by referencing the correlation matrix, and only selecting features that had high feature importances for model performance.

We spent some time investigating different model metrics that would be appropriate for our project's pipeline and measure results.
- discuss MAPE, percent classsified correctly within margin
-


# Special Thanks
- [UCSD Halıcıoğlu Data Science Institute](https://datascience.ucsd.edu/)
- [Viasat](https://www.viasat.com/)
- [DANE](https://github.com/dane-tool/dane)
