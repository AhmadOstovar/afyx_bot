# AFRY X Bot

The AFRY X Bot is a program that answers questions based on a set of documents.

## Table of contents
1. [Description](#description)
2. [Deployment](#deployment)
3. [Use](#use)

<a id="description" /></a>
## 1. Description

<a id="large-language-models-and-prompt-engineering" /></a>
### Large language models and prompt engineering
AFRY X Bot uses prompt engineering to answer questions from a given set of documents. Large language models (LLMs) can be used to answer questions: you write a question and the model generates an answer. In prompt engineering you give an LLM some text as a context, and ask the model to answer a question based on that context. Based on [this](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb) example, AFRY X Bot uses the following steps to achieve this:

1. Take a collection of articles or documents and create a numerical representation for each of these documents, i.e., an embedding. Save these embeddings.
2. Take a question from the user and for this question create an embedding using the same program that was used to create the document embeddings in step 1.
3. Calculate which document embeddings (step 1) most closely resemble the question embedding (step 2).
4. Send the text of the most closely resembling documents to the LLM and add your question at the end. The model will now use your prompt to answer your question.

This question answering is currently built on [OpenAI](https://openai.com/)'s [ChatGPT](https://chat.openai.com/). However, other LLMs may be used after some editing of the code. The `text-davinci-003` model at the core of ChatGPT is the latest GPT-3-based model on OpenAI. OpenAI also hosts other, older models that work faster and may also perform good enough. Click [here](https://beta.openai.com/docs/models/gpt-3) for more information.

For embeddings we use the second generation embedding model `text-embedding-ada-002`, which is currently the best method on OpenAI. This embedding model creates embeddings of length 1536. Click [here](https://beta.openai.com/docs/guides/embeddings/what-are-embeddings) for more information.

When creating the embeddings the text needs to be tokenized, and the `text-embedding-ada-002` uses the tokenizer `cl100k_base`. This tokenization for GPT-3 is the same as for GPT-2. Click [here](https://stackoverflow.com/a/74875104) for more information.

### Chatbot
We recently added the chatbot function. This needs more documentation.

### Functionality
We tried to adhere to MLOps principles during development of this repositories. This means that we distinguish between a data loop and a model loop.

1. **Model loop**: Because the code uses the OpenAI API, there is no modeling. Therefore the model loop only concerns calling a deployed model's API through an Azure Function App function. The API is either called through the function `run_completer` or `run_chatbot`.
2. **Data loop**: The data loop is separated into four steps, and each step will have its own storage folder:
    1. source: the source data that is migrated
    2. validated: raw source data after validation
    3. cleaned: cleaned data as {"title": {"ref": url-like-reference, "content": some-text}, ...}
    4. curated: curated data split into files `records.json` and `embeddings.json`
        - `records.json`: {"title": {"ref": url-like-reference, "content": some-text, "tokens": number-of-tokens}, ...}
        - `embeddings.json`: {"title": [...], ...}
    
    At the time of writing only the curation step is available through the function `curate`. Other functions and generalizations may be added at a later stage.

<a id="deployment" /></a>
## 2. Deployment

The application uses the OpenAI API. You will be charged for its use, but during testing these charges are unlikely to go above one dollar. For more information on OpenAI pricing, click [here](https://openai.com/api/pricing/).

### Cloud
The code is intended for deployment through the Azure Function App. An accompanying Terraform repository will take care of this deployment. This deployment uses a Docker image, which can be created [following these steps](#docker-image).

We followed [this](https://learn.microsoft.com/en-us/azure/azure-functions/functions-create-function-linux-custom-image?tabs=in-process%2Cbash%2Cazure-cli&pivots=programming-language-python) tutorial when creating our containerized deployment.

### Local
Azure Function Apps can be developed and tested in a local environment. This means we can deploy our Function App locally. However, you must have an Azure Storage Account. To do so, follow the steps below.

1. [Install the Azure Functions Core Tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local?tabs=v4%2Clinux%2Ccsharp%2Cportal%2Cbash#v2).

#### Linux

2. Create a virtual environment. Using conda, for example

    conda create -n afryxbot_env python=3.9
 
3. Install the requirements file

    pip3 install -r requirements.txt

4. Set your environment to automatically create an environment variable named `OPENAI_KEY` with your OpenAI API key. This key needs to be generated on the OpenAI account you want to use for this image. Detailed instructions for conda can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux). In brief, first execute these bash commands

        cd $CONDA_PREFIX
        mkdir -p ./etc/conda/activate.d
        mkdir -p ./etc/conda/deactivate.d
        touch ./etc/conda/activate.d/env_vars.sh
        touch ./etc/conda/deactivate.d/env_vars.sh

    Next, edit the file `./etc/conda/activate.d/env_vars.sh` by adding the line

        export OPENAI_KEY="<sk-...>"

    and edit the file `./etc/conda/deactivate.d/env_vars.sh` by adding the line

        unset OPENAI_KEY

5. Activate your environment

        conda activate afryxbot_env

#### Windows

2. Create a virtual environment, if you use [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) you can create it with this command:

    py -m venv bot_env

3. Install the requirements file

    pip3 install -r requirements.txt

4. Set your environment to automatically create an environment variable named `OPENAI_KEY` with your OpenAI API key. To do this, add this line to ./bot_env/Scripts/Activate.ps1:

    $Env:OPENAI_KEY = "<sk-...>"

5. Activate your environment

    ./bot_env/Scripts/Activate.ps1

6. In you storage account create a container named `data`. That container needs two sub-directories `cleaned` and `curated`.

6. Open or create the file `local.settings.json` in your repository's root directory with the following content

        {
            "IsEncrypted": false,
            "Values": {
                "FUNCTIONS_WORKER_RUNTIME": "python",
                "AzureWebJobsStorage": "CONNECTIONSTRING"
            }
        }
    
    Change the AzureWebJobsStorage to the connection string of your storage account.

7. Open the file `./run_completer/function.json` and on line 5 change

        "authLevel": "Function",
    
    to

        "authLevel": "anonymous",
    
    *We are only doing this for local deployment, bypassing the need of a key. Some form of access control is recommended for online deployment!*

7.1. Upload your data to the blob

8. Use the following command to start your Azure Function App

        func start

<a id="docker-image" /></a>
### Docker image

Before you start, make sure you have Docker installed on your machine. If you need to install, follow the instructions [here](https://docs.docker.com/engine/install/).

Next, make sure docker is running. For bash the command is

    sudo service docker start

With docker running, navigate to the repository's root directory. For a general image we use the following command to create a docker image

    docker build --build-arg "OPENAI_API_KEY=<sk-...>" .

The parameter `OPENAI_KEY` creates a environment variable in your docker container with the OpenAI API key. This key needs to be generated on the OpenAI account you want to use for this image.

When deploying AFRY X Bot on Azure, however, we store our Docker image on the Azure Container Registry. To streamline this operation we add a `--tag` to the command

    docker build --build-arg "OPENAI_API_KEY=<sk-...>" --tag <containeregistryname>.azurecr.io/<imagename>:v1.0.0 .

Parameter `containeregistryname` is the name of your Azure Container Registry. The `imagename` is the name you want for your image and is followed by a version number. You increment the version number during the life cycle of your deployment.

<a id="use" /></a>
## 3. Use

After deployment we can send our requests to the server. Parameters that determine the exact behavior of our app are set in `settings/settings.json` and discussed in more detail in the documentation linked in **[Large language models and prompt engineering](#large-language-models-and-prompt-engineering)**. We first process data and next query our model.

### Data curating
Copy the file `records_clean.json` from the folder `./mockup_data` in this repository to the `data/cleaned` folder in your storage account. This operation will trigger a run of the `curate` function and create the files `records.json` and `embeddings.json` in the folder `data/curated` on the storage account.

### Querying
With your data files created you can now use the `run_completer` function. Without a front end we need to manually adjust the http-request.

#### Cloud
We need the function key to get access. Open your Function App in the Azure portal, click on `Functions`, click on the function `run_completer`, and click on `Function Keys`. From there, copy the `default` key value. We can now send a first request for an answer through the command below. Make sure to paste the `default` key value to `code=<...>` and change the `<app-name>` to the name of your Azure Function App. Feel free to change the `query=<...>` to something of your liking!

    https://<app-name>.azurewebsites.net/api/run_completer?code=<...>&query="What should I do when I start my new job?"

#### Local
Because we disabled the key we do not need to provide a `code=<...>`. This means we could send a first request for an answer through the command below. Feel free to change the `query=<...>` to something of your liking!

    http://localhost:7071/api/run_completer?query="What should I do when I start my new job?"