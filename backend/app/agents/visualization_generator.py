from agno.agent import Agent
import pandas as pd
import json

class VisualizationGenerator(Agent):
    def __init__(self):
        super().__init__(name="VisualizationGenerator")

    async def execute(self, data):
        """
        Generate visualization data from prediction results
        """
        try:
            prediction_results = data.get("prediction_results", {})
            cleaned_data = data.get("cleaned_data", [])
            
            print(f"VisualizationGenerator received prediction_results: {list(prediction_results.keys())}")
            
            # Prepare visualization data
            visualization_data = {
                "prediction_results": prediction_results,
                "cleaned_data": cleaned_data,
                "chart_data": {
                    "predictions_vs_actual": {
                        "predictions": prediction_results.get("predictions", []),
                        "actual_values": prediction_results.get("actual_values", [])
                    },
                    "model_info": {
                        "model_name": prediction_results.get("model", "Unknown"),
                        "target_variable": prediction_results.get("target_variable", "Unknown"),
                        "feature_variables": prediction_results.get("feature_variables", []),
                        "mse": prediction_results.get("mean_squared_error", 0)
                    }
                }
            }
            
            return visualization_data
        except Exception as e:
            print(f"Error in VisualizationGenerator: {str(e)}")
            import traceback
            traceback.print_exc()
            raise