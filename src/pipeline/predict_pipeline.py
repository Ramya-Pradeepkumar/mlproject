
import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


# Find the project root independently of the current working directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join(
    PROJECT_ROOT, "artifacts", "proprocessor.pkl"
)


class PredictPipeline:
    def __init__(self):
        try:
            self.model = load_object(file_path=MODEL_PATH)
            self.preprocessor = load_object(
                file_path=PREPROCESSOR_PATH
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, features):
        try:
            transformed_data = self.preprocessor.transform(features)
            predictions = self.model.predict(transformed_data)
            return predictions

        except Exception as e:
            raise CustomException(e, sys) from e


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [
                    self.parental_level_of_education
                ],
                "lunch": [self.lunch],
                "test_preparation_course": [
                    self.test_preparation_course
                ],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys) from e
