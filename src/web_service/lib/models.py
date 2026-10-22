# Pydantic models for the web service

from typing import Literal

from pydantic import BaseModel, Field


class AbaloneInput(BaseModel):
    """Measurements of one abalone (same columns as the dataset)."""

    # Sex can only be M, F or I (infant), anything else is refused with a 422 error
    sex: Literal["M", "F", "I"]
    # Field(gt=0): the measurements must be greater than 0
    length: float = Field(gt=0)
    diameter: float = Field(gt=0)
    height: float = Field(gt=0)
    whole_weight: float = Field(gt=0)
    shucked_weight: float = Field(gt=0)
    viscera_weight: float = Field(gt=0)
    shell_weight: float = Field(gt=0)


class PredictionOutput(BaseModel):
    """Predicted number of rings, and the age (rings + 1.5)."""

    rings: float
    age: float
