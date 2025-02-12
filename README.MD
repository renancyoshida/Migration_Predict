# Estimating a migration model

## Motivation

This is a ML task that is part of a chapter in my PhD dissertation. In Brazil, landless people occupy farms to obtain land titles. The objective here is: conditional on an occupation, can we predict where the occupiers are from? The task is similar to a migration model where each sample is an origin-destination combination and the outcome is a binary of whether there was any movement in that pair. 

Below are two maps. The first displays in blue occupations where I have people's origins and in red the ones I do not, hence the issue. The red polygons are then the production dataset, and the blue the labeled data.

<img src="https://github.com/renancyoshida/Migration_Predict/blob/master/migration/figures/occ-match-map.png?raw=true" width="600">

The second map displays an example. A red dot where occupation took place and the shaded polygons are origins.

<img src="https://github.com/renancyoshida/Migration_Predict/blob/master/migration/figures/occ-map-maisa.png?raw=true" width="600">

## Data

I use demographic and economic variables from both origin and destination such as income, population, literacy rate, urbanization. I also include distance between pairs, travel time, a dummy for whether there is road access, and a dummy for same city (they often occupy lands in the same city where they are from).


## Methodology

* Feature engineering: I logarize variables that appear to have log-normal distribution and include features I believe important here: distance, travel time and a dummy for being in the same city.

* Evaluation: Given large class imbalance, I use AUC_PR. 

* Models: I compare a baseline logistic regression with an out-of-the-box XGBoost and a fine-tuned XGBoost. For each model, a Cross-Validation with Grid Search to optimize hyperparameters was conducted.

## Results

* XGBoost and tuning exhibit a substantial improvement:

<img src="https://github.com/renancyoshida/Migration_Predict/blob/master/migration/figures/pr_curve.png?raw=true" width="600">

* The final metric is still low, but this is a hard problem to solve and a work-in-progress. This repo will be updated in the next few montths.

* I also evaluate feature importance: same_muni and population in origin city are the most important predictors:

<img src="https://github.com/renancyoshida/Migration_Predict/blob/master/migration/figures/feature_importance.png?raw=true" width="600">
