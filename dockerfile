# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy necessary files into the container
COPY museum_lang_api.py .
COPY language_detection_pipeline.pkl .

# 4. Install required dependencies
RUN pip install fastapi uvicorn scikit-learn numpy pydantic

# 5. Expose the application's port
EXPOSE 8000

# 6. Define the default command to run the app
CMD ["uvicorn", "museum_lang_api:app", "--host", "0.0.0.0", "--port", "8000"]
