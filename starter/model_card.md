# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random forest classifier with default setting trained on the census data for classification task

## Intended Use
This model leverages features such as workclass, education etc to classify the salary level.

## Training Data
Training data is 80% of the total input census data

## Evaluation Data
Evaluation data is the remaining 20% of the total input census data
## Metrics
Three different eval metrics are being used.
- Precision: 0.73
- Recall: 0.65
- Fbeta: 0.69.

## Ethical Considerations
We conducted model slices to ensure model not bias towards a specific education group, race or workclass etc


## Caveats and Recommendations
Additional hyperparameter tuning, as well as additional feature engineering could be incorporated to improve model performances