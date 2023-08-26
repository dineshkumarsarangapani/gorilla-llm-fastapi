# gorilla-llm-fastapi

Step1: docker build -t gorilla .
Step2: docker run -p 80:80 --gpus all gorilla -e AZUREML_MODEL_DIR=/home/azureuser/cloudfiles/code/gorilla-falcon-7b-hf-v0