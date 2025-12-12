from agno.agent import Agent
import google.generativeai as genai
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json
import re

class PredictionAgent(Agent):
    def __init__(self, api_key):
        super().__init__(name="PredictionAgent")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    async def execute(self, data):
        try:
            cleaned_data = data["cleaned_data"]
            user_query = data["user_query"]
            df = pd.DataFrame(cleaned_data)
            
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {list(df.columns)}")

            # Use Gemini API to identify features and target
            prompt = f"""
            Given the dataset columns: {list(df.columns)}
            And the user query: "{user_query}"
            
            Identify the target variable for prediction and the relevant feature variables.
            Respond with ONLY a JSON object (no markdown, no extra text) containing "target_variable" and "feature_variables".
            
            Example response format:
            {{"target_variable": "price", "feature_variables": ["area", "bedrooms"]}}
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            print(f"Gemini response: {response_text}")
            
            # Clean up the response to extract JSON
            # Remove markdown code blocks if present
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*', '', response_text)
            response_text = response_text.strip()
            
            try:
                response_json = json.loads(response_text)
                target_variable = response_json["target_variable"]
                feature_variables = response_json["feature_variables"]
                
                # Validate that columns exist
                if target_variable not in df.columns:
                    raise ValueError(f"Target variable '{target_variable}' not found in dataset")
                
                missing_features = [f for f in feature_variables if f not in df.columns]
                if missing_features:
                    raise ValueError(f"Feature variables not found: {missing_features}")
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error parsing Gemini response or validating columns: {e}")
                # Fallback: use last column as target, rest as features
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) < 2:
                    raise ValueError("Not enough numeric columns for prediction")
                target_variable = numeric_cols[-1]
                feature_variables = numeric_cols[:-1]
                print(f"Using fallback - Target: {target_variable}, Features: {feature_variables}")

            # Prepare data for model
            X = df[feature_variables].copy()
            y = df[target_variable].copy()

            # Handle any remaining missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

            # One-hot encode categorical features
            X = pd.get_dummies(X, drop_first=True)
            
            # Check if we have enough data
            if len(X) < 2:
                raise ValueError("Not enough data points for train-test split")

            # Adjust test_size based on dataset size
            test_size = min(0.2, max(0.1, 1 / len(X)))
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            return {
                "prediction_results": {
                    "model": "Linear Regression",
                    "target_variable": target_variable,
                    "feature_variables": feature_variables,
                    "mean_squared_error": float(mse),
                    "predictions": predictions.tolist(),
                    "actual_values": y_test.tolist()
                },
                "cleaned_data": cleaned_data
            }
        except Exception as e:
            print(f"Error in PredictionAgent: {str(e)}")
            import traceback
            traceback.print_exc()
            raise