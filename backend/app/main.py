from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.services.file_parser import parse_file
from app.agents.data_analyzer import DataAnalyzer
from app.agents.data_cleaner import DataCleaner
from app.agents.pca_performer import PCAPerformer
from app.agents.prediction_agent import PredictionAgent
from app.agents.visualization_generator import VisualizationGenerator
from app.models.data_models import AnalysisResult, PredictionPayload
import os

app = FastAPI()

# It's recommended to load the API key from an environment variable for security
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Configure CORS to allow communication with the frontend
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_data(file: UploadFile = File(...)):
    try:
        df = await parse_file(file)
        
        # Initialize agents
        data_analyzer = DataAnalyzer()
        data_cleaner = DataCleaner()
        pca_performer = PCAPerformer()

        # Manual orchestration - run agents sequentially
        # Step 1: Analyze the data
        analysis_data = {"dataframe": df}
        analyzer_result = await data_analyzer.execute(analysis_data)
        
        # Step 2: Clean the data
        cleaner_result = await data_cleaner.execute(analyzer_result)
        
        # Step 3: Perform PCA
        pca_result = await pca_performer.execute(cleaner_result)
        
        # Combine results for response
        final_result = {
            "total_rows": analyzer_result["total_rows"],
            "total_columns": analyzer_result["total_columns"],
            "is_clean": analyzer_result["is_clean"],
            "noise_types": analyzer_result["noise_types"],
            "cleaned_data": pca_result["cleaned_data"],
            "pca_result": pca_result["pca_result"]
        }

        return final_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(payload: PredictionPayload):
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini API key not set.")

        # Initialize agents
        prediction_agent = PredictionAgent(api_key=GEMINI_API_KEY)
        visualization_generator = VisualizationGenerator()

        # Manual orchestration
        # Step 1: Make predictions
        prediction_data = {
            "cleaned_data": payload.cleaned_data,
            "user_query": payload.user_query
        }
        
        print(f"Making predictions with query: {payload.user_query}")
        prediction_result = await prediction_agent.execute(prediction_data)
        
        print(f"Prediction result keys: {prediction_result.keys()}")
        
        # Step 2: Generate visualizations
        visualization_result = await visualization_generator.execute(prediction_result)

        return visualization_result
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)