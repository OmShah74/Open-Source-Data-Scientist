from pydantic import BaseModel
from typing import List, Dict, Any

class AnalysisResult(BaseModel):
    total_rows: int
    total_columns: int
    is_clean: bool
    noise_types: List[str]
    cleaned_data: List[Dict[str, Any]]
    pca_result: Dict[str, Any]

class PredictionPayload(BaseModel):
    cleaned_data: List[Dict[str, Any]]
    user_query: str