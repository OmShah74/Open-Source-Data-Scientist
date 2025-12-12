from agno.agent import Agent
import pandas as pd

class DataAnalyzer(Agent):
    def __init__(self):
        super().__init__(name="DataAnalyzer")

    async def execute(self, data):
        df = data["dataframe"]
        total_rows, total_columns = df.shape
        
        # Check for missing values and duplicates
        is_clean = df.isnull().sum().sum() == 0 and not df.duplicated().any()
        
        noise_types = []
        if df.isnull().sum().any():
            noise_types.append("Missing Values")
        if df.duplicated().any():
            noise_types.append("Duplicate Rows")

        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "is_clean": is_clean,
            "noise_types": noise_types,
            "original_dataframe": df
        }