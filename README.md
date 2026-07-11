# Student Performance Prediction

This project predicts a student's maths score based on things like gender, parental education, lunch type, test preparation, and their reading and writing scores. It goes from raw data all the way to a working web app where you can type in a student's details and get a predicted score back.

## What is in this project

- **Exploratory analysis** (`src/notebook/EDA_studentperformance.ipynb`): looks at the dataset, checks for missing values, and explores which factors seem to affect scores the most.
- **Model comparison notebook** (`src/notebook/ML_model.ipynb`): builds a preprocessing pipeline and compares 10 different regression models to find the one that predicts maths scores best.
- **Production pipeline** (`src/components/`): the same steps from the notebooks, rebuilt as proper Python components that can run on their own.
  - `data_ingestion.py`: reads the raw data and splits it into training and test sets.
  - `data_transformation.py`: scales numeric columns and encodes categorical columns.
  - `model_trainer.py`: trains several models, tunes their settings, and saves the best one.
- **Prediction pipeline** (`src/components/pipeline/predict_pipeline.py`): loads the saved model and preprocessor so new student data can be scored.
- **Web app** (`app.py`): a small Flask app with a form where you fill in a student's details and get a predicted maths score.

## Key finding

Out of 10 models tested (Linear Regression, Ridge, Lasso, KNN, Decision Tree, Random Forest, AdaBoost, SVR, XGBoost, CatBoost), the simplest models, Ridge and Lasso, performed best on unseen data, explaining about 88% of the variation in maths scores. The more complex models like Random Forest and XGBoost actually overfit the training data and did worse on the test set.

## Dataset

The dataset (`src/notebook/data/student.csv`) has 1000 student records with these columns: gender, race/ethnicity, parental level of education, lunch type, test preparation course, and scores in maths, reading, and writing.

## How to run it

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the training pipeline (this reads the data, builds the preprocessing pipeline, trains the models, and saves the best one to `artifacts/`):
   ```
   python -m src.components.data_ingestion
   ```

3. Start the web app:
   ```
   python app.py
   ```
   Then open `http://127.0.0.1:5000` in your browser, fill in the form, and get a predicted maths score.

## Tech used

Python, pandas, scikit-learn, XGBoost, CatBoost, Flask.
