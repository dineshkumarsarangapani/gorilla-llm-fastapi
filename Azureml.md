export  ACR_NAME='4c3bbcb706144dbeb4a9db8d390ced0a'
export  IMAGE_TAG='gorilla-llm-falcon-hf'
export  BASE_PATH='.'
export  ENDPOINT_NAME='gorilla-llm-falcon-hf'

az acr login -n $ACR_NAME
az acr build -t $IMAGE_TAG -f $BASE_PATH/Dockerfile -r $ACR_NAME $BASE_PATH

az ml online-endpoint create -n $ENDPOINT_NAME

# <check_endpoint_status> 
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else 
  echo "Endpoint creation failed"
  exit 1
fi
# </check_endpoint_status> 



# <create_deployment> 
change_vars $BASE_PATH/ts-hf-tg-deployment.yml
az ml online-deployment create -f $BASE_PATH/ts-hf-tg-deployment.yml_ --all-traffic
# </create_deployment> 

# <check_deployment_status> 
deploy_status=`az ml online-deployment show --endpoint-name $ENDPOINT_NAME --name gorilla-llm-hf-falcon --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
    echo "Deployment completed successfully"
else
    echo "Deployment failed"
    exit 1
fi
# </check_deployment_status> 

# <get_endpoint_details> 
# Get key
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv)

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"
# </get_endpoint_details> 

# <test_endpoint> 
curl -X POST -H "Authorization: Bearer $KEY" -T "$SERVE_PATH/Text_gen_artifacts/sample_text.txt" $SCORING_URL
# </test_endpoint> 