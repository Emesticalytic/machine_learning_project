import os
import sys
from dataclasses import dataclass

import numpy as np

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Random Forest Regressor': RandomForestRegressor(random_state=42),
                'AdaBoost Regressor': AdaBoostRegressor(random_state=42),
                'SVR': SVR(),
                'XGBRegressor': XGBRegressor(random_state=42, verbosity=0),
                'CatBoosting Regressor': CatBoostRegressor(random_state=42, verbose=False),
            }

            params = {
                'Linear Regression': {},
                'Ridge': {
                    'alpha': [0.01, 0.1, 1, 10, 100],
                },
                'Lasso': {
                    'alpha': [0.01, 0.1, 1, 10, 100],
                },
                'K-Neighbors Regressor': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                },
                'Decision Tree': {
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],
                    'max_depth': [3, 5, 7, 10, None],
                },
                'Random Forest Regressor': {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                'AdaBoost Regressor': {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                'SVR': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                },
                'XGBRegressor': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                'CatBoosting Regressor': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=lambda name: model_report[name])
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found', sys)

            logging.info(f'Best found model on training and testing dataset: {best_model_name}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return best_model_name, r2_square
        except Exception as e:
            raise CustomException(e, sys)
