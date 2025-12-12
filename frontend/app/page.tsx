'use client';

import { useState } from 'react';
import axios from 'axios';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [predictionQuery, setPredictionQuery] = useState('');
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post('http://localhost:8000/api/analyze', formData);
      setAnalysis(res.data);
    } catch (err) {
      setError('Analysis failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!analysis || !predictionQuery) return;
    setIsLoading(true);
    setError(null);

    try {
      const res = await axios.post('http://localhost:8000/api/predict', {
        cleaned_data: analysis.cleaned_data,
        user_query: predictionQuery,
      });
      setPredictionResult(res.data);
    } catch (err) {
      setError('Prediction failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderVisualization = (viz: any, index: number) => {
    if (viz.type === 'scatter') {
      return (
        <div key={index} className="mt-6">
          <h3 className="text-xl font-semibold mb-2">{viz.title}</h3>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart>
              <CartesianGrid />
              <XAxis type="number" dataKey="x" name={viz.x_label} />
              <YAxis type="number" dataKey="y" name={viz.y_label} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              <Scatter name="Data" data={viz.data} fill="#8884d8" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <header className="text-center mb-12">
        <h1 className="text-5xl font-bold">AI Data Scientist</h1>
        <p className="text-xl text-gray-400 mt-2">Your personal data science assistant.</p>
      </header>

      <main className="max-w-4xl mx-auto">
        {/* File Upload */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8">
          <h2 className="text-2xl font-semibold mb-4">1. Upload Data</h2>
          <div className="flex items-center space-x-4">
            <label htmlFor="dataFile" className="sr-only">Upload data file</label>
            <input 
              id="dataFile"
              type="file" 
              onChange={handleFileChange} 
              className="file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:bg-violet-50 file:text-violet-700 hover:file:bg-violet-100"
              aria-label="Upload data file"
            />
            <button onClick={handleAnalyze} disabled={!file || isLoading} className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-bold py-2 px-4 rounded-full">
              {isLoading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        </div>

        {/* Analysis Results */}
        {analysis && (
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8">
            <h2 className="text-2xl font-semibold mb-4">2. Analysis Results</h2>
            <div className="grid grid-cols-2 gap-4">
              <p>Total Rows: {analysis.total_rows}</p>
              <p>Total Columns: {analysis.total_columns}</p>
              <p>Is Clean: {analysis.is_clean ? 'Yes' : 'No'}</p>
              {!analysis.is_clean && <p>Noise: {analysis.noise_types.join(', ')}</p>}
            </div>
            {analysis.pca_result?.principal_components && (
              <div className="mt-6">
                <h3 className="text-xl font-semibold mb-2">PCA</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart>
                    <CartesianGrid />
                    <XAxis type="number" dataKey="0" name="PC1" />
                    <YAxis type="number" dataKey="1" name="PC2" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="PCA" data={analysis.pca_result.principal_components} fill="#8884d8" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {/* Prediction */}
        {analysis && (
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8">
            <h2 className="text-2xl font-semibold mb-4">3. Make a Prediction</h2>
            <div className="flex flex-col space-y-4">
              <input type="text" value={predictionQuery} onChange={(e) => setPredictionQuery(e.target.value)} placeholder="e.g., 'Predict salary based on experience'" className="bg-gray-700 border border-gray-600 rounded-lg py-2 px-4 focus:outline-none" />
              <button onClick={handlePredict} disabled={!predictionQuery || isLoading} className="bg-green-600 hover:bg-green-700 disabled:bg-green-400 text-white font-bold py-2 px-4 rounded-full self-start">
                {isLoading ? 'Predicting...' : 'Predict'}
              </button>
            </div>
          </div>
        )}

        {/* Prediction Results */}
        {predictionResult && (
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
            <h2 className="text-2xl font-semibold mb-4">4. Prediction & Visualization</h2>
            <div>
              <p>Model: {predictionResult.prediction_results.model}</p>
              <p>Target: {predictionResult.prediction_results.target_variable}</p>
              <p>MSE: {predictionResult.prediction_results.mean_squared_error.toFixed(2)}</p>
            </div>
            {predictionResult.visualizations.map(renderVisualization)}
          </div>
        )}

        {error && (
          <div className="bg-red-800 text-white p-4 rounded-lg mt-4">
            {error}
          </div>
        )}
      </main>
    </div>
  );
}