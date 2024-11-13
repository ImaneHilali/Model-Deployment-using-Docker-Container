## Cloud Run Deployment for Multilingual File Translation using NLLB Model
This repository contains a Dockerized Flask application deployed on Google Cloud Run that leverages the Facebook NLLB (No Language Left Behind) model for multilingual translation. It listens to Cloud Storage events, processes incoming text files, translates the content into multiple languages, and uploads the translated files back to Cloud Storage. The system supports automatic translation between English, Spanish, and French based on detected language, making it ideal for scalable and serverless translation tasks. The code includes Dockerfile configuration, necessary dependencies in requirements.txt, and translation logic in translation_script.py
