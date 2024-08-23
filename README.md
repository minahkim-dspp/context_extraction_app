# Extracting Geographical Context from Texts Describing Farming Practices :earth_africa:
This repository is a partial replica of the private repository from IFPRI (@IFPRI). Its main purpose is to serve as a portfolio. It includes the code that I wrote during my second half of the internship (February 2024 - May 2024) as a part of the GAIA (Generative AI for Agricultural Development) project.

## Set Up
### API Token Requirement
You will need the **[OpenAI API Key](https://openai.com/api/)**, **[Huggingface Access Token](https://huggingface.co/docs/hub/security-tokens)**, and the **[Pinecone API Key](https://docs.pinecone.io/guides/getting-started/authentication)** to successfully run the codes. Please visit the website to learn more about attaining the API tokens. We encourage the users to set up the `.env` file in a following manner:
```
OPENAI_API_KEY = [Insert your OpenAI API Key Here]
HUGGINGFACE_API_TOKEN = [Insert your Huggingface Access Token Here]
PINECONE_API_KEY = [Insert your Pinecone API Key Here]
```
### Comparison with AgriNER (Agriculture Named Entity Recognition)
The context_extraction_app/test.ipynb has a section that demonstrates the performance of the Open Research Knowledge Graph Agriculture Named Entity Recognition (the ORKG Agri-NER) model by Jennifer D'Souza and Omar Arab-Oghli. In order to activate this portion, you need to install the ORKG Agri-NER model in your device. Then, edit the constant called `NER_MODEL_PATH` on line 19 of context_extraction_app/entity_extraction.py. Here are the resources that I used when installing the ORKG Agri-NER model.

- README: https://gitlab.com/TIBHannover/orkg/nlp/experiments/orkg-agriculture-ner/-/raw/v0.1.0/README.md
- Relevant Paper: https://doi.org/10.3390/knowledge4010001

  
## How to Navigate This Repository
This repository consists of two folders: context_extraction_app and document. 
- **context_extraction_app** 
  - **test.ipynb** : This jupyter notebook includes the most high level codes and visualizes the result.
  - document.py: This module contains the Document object that keeps the text and initiates the geographical context extraction.
  - text_preprocessing.py: This module parses the text from the pdf and splits it into examinable tokens.
  - entity_extraction.py: This module includes codes that extract geographical location in texts using AgriNER and RAG. 
  - openai_call.py : This module hosts a function that calls OpenAI API. 
  - pinecone_setup.py: This module consists all codes that assist the interaction with Pinecone vector storage.
- **document** : This folder holds the text used to test the models.

## Relevant Link
:page_facing_up: [Blog post](https://minahkim.georgetown.domains/applying-generative-ai-in-food-policy/) summarizing my work at IFPRI to apply Generative AI in food policy. This post includes the description of this project.

## Contact
For further insturction or questions, please reach out to MinAh Kim (mk2215@georgetown.edu / minahkim.official@gmail.com)
