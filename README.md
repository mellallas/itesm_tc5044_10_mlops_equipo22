# Train the model
conda activate [env]
python models\train_model.py

# Build Docker image
docker build -t steelindustry-classification-api

# Run Docker container
docker run -b 8000:8000 steelindustry-classification-api

# Test the API
curl -Method Post -Uri "http://localhost:8000/predict" -Headers @{ "Content-Type" = "application/json" }
-Body '{"features": [3.17, 2.95, 0.0, 0.0, 73.21, 100.0, 900, Weekday, Monday]}'