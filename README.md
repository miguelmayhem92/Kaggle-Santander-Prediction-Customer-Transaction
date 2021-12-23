# Kaggle-Santander-Prediction-Customer-Transaction
Kaggle link: https://www.kaggle.com/c/santander-customer-transaction-prediction

### Brief description of the challenge

Predict customer transaction (a binary target) based on middle size data. The train data has about 200 features and 300K rows
Some particularities of the data:
* Imbalanced data
* cloudy data meaning that there is no clear patern to diffirenciate the target

### My strategy to tackle the challenge:
* Exploration
* basic feature selection
* feature generation using mahalanobis distance and counting encoding 
* Data augmentation and data reduction strategy
* Models explorations: KNN, RF and GBM
* Model Selection (using Kfold cross validation and some tunning)
* Furhter tunning
* Prediction

#### Notebooks:
* [Exploration](https://github.com/miguelmayhem92/Kaggle-Santander-Prediction-Customer-Transaction/blob/main/CST_explo.ipynb)
* [Modeling](https://github.com/miguelmayhem92/Kaggle-Santander-Prediction-Customer-Transaction/blob/main/CST_modeling.ipynb)
* [Tunning](https://github.com/miguelmayhem92/Kaggle_Santander/blob/main/CST_Tunning.ipynb)
* [Tunning Random Forest](https://github.com/miguelmayhem92/Kaggle_Santander/blob/main/CST_RF_Tunning.ipynb)
