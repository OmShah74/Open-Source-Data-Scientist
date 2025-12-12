from agno.agent import Agent
import pandas as pd

class DataCleaner(Agent):
    def __init__(self):
        super().__init__(name="DataCleaner")

    async def execute(self, data):
        df = data["original_dataframe"].copy()  # Create a copy to avoid SettingWithCopyWarning
        
        # Impute missing values for numeric columns with the mean
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
            
        # Remove duplicate rows
        df = df.drop_duplicates()

        return {
            "cleaned_dataframe": df
        }