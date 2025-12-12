from agno.agent import Agent
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

class PCAPerformer(Agent):
    def __init__(self):
        super().__init__(name="PCAPerformer")

    async def execute(self, data):
        df = data["cleaned_dataframe"]
        
        numeric_df = df.select_dtypes(include=['number'])
        
        pca_result = {}
        if not numeric_df.empty:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            
            pca = PCA(n_components=2)  # Reduce to 2D for visualization
            principal_components = pca.fit_transform(scaled_data)
            
            pca_result = {
                "principal_components": principal_components.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist()
            }

        return {
            "cleaned_data": df.to_dict(orient="records"),
            "pca_result": pca_result
        }