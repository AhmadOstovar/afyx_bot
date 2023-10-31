# To enable ssh & remote debugging on app service change the base image to the one below
# FROM mcr.microsoft.com/azure-functions/python:4-python3.10-appservice
FROM mcr.microsoft.com/azure-functions/python:4-python3.10

ARG OPENAI_API_KEY

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true \
    OPENAI_KEY=$OPENAI_API_KEY

COPY requirements.txt /
RUN pip install -r /requirements.txt

RUN python -c "import nltk; nltk.download('punkt')"

COPY . /home/site/wwwroot