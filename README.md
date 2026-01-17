# Neumann's Matrix — Round 2

## BNA26 Computer Science Module

Before you begin, make a backup copy of this entire folder.

This repository contains a machine learning project with intentional mistakes. Your job is to find and fix them to achieve the best model accuracy.

## What You'll Find Here

The repository is organized as follows:

- `data/dataset.csv` — The dataset containing system metrics and power consumption data
- `src/data_analysis.ipynb` — A notebook with visualizations to help you understand the data
- `src/models.ipynb` — The main notebook where you'll fix the mistakes
- `models/` — Where your trained models and metrics will be saved after running

The dataset has one target column (`power_consumption`) that we're trying to predict, and ten feature columns representing various system metrics like CPU utilization, memory usage, temperature, etc.

## What You Need To Do

There are two types of mistakes in `models.ipynb` that you need to fix:

1. The wrong feature columns are being used for training
2. The model hyperparameters are set to poor values

Start by opening `src/data_analysis.ipynb` and running all the cells. This will generate several graphs showing how each feature relates to the target variable. Use your knowledge and understanding of different graphs to find what features are strongly correlated with the target variable and which aren't.

Once you've identified which features matter most, open `src/models.ipynb`. Near the top, you'll find a line that sets `feature_columns`. Replace the current values with the features you identified as most relevant.

Next, go through each of the five models in the notebook. Each one has comments marked with `TODO` that point out what's wrong with the hyperparameters. The comments also include hints about what values might work better. Use your understanding of machine learning to choose appropriate values.

The models you'll be working with are:

- Polynomial Regression
- Random Forest Regression  
- MLP (Neural Network) Regression
- Support Vector Regression
- Elastic Net Regression

After making your changes, click "Run All" at the top of the notebook. This will train all models and save them to the `models/` folder. Check the printed metrics for each model — a higher R² score (closer to 1.0) and lower MSE/MAE values indicate better performance.

## Submission

Once you're satisfied with your results:

1. Find the `models/` folder in this repository
2. Compress it into a zip file
3. Rename the zip file to your delegation name (e.g., `YourDelegation.zip`)
4. Submit it to: [SUBMISSION LINK HERE]

## Setup

If you need to set up the environment again:

```bash
git clone <repository-url>
cd BNA26-ML
pip install -r requirements.txt
```

You'll need Python 3.8+ and either Jupyter Notebook or VS Code with the Jupyter extension.

## Evaluation

Your submission will be judged on the accuracy of your trained models, metrics are also printed under each model you run above the graph. Better feature selection and hyperparameter tuning will result in bigger R2 and lower MSE and MAE.

You're allowed to modify anything in `models.ipynb`, not just the TODO sections. You can even add new models if you want. The goal is simply to achieve the best possible accuracy.

## Help and guidance

If you run into any problems or want to ask any questions regarding what you don't understand, ask the host team freely, they are there to guide you in all issues other than giving the answers.
