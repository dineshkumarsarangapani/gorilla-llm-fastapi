# gorilla-llm-fastapi

Step1: docker build -t gorilla .
Step2: docker run  -p 8032:80 -v /home/azureuser/cloudfiles/code/gorilla-falcon-7b-hf-v0/:/model_path -e AZUREML_MODEL_DIR='/model_path' --gpus all gorilla
