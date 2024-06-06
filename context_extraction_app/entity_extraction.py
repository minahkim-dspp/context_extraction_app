##########################################################
### Entity_Extraction
### This module encapsulates the entity extraction process
##########################################################

import pandas as pd 
import numpy as np
import re

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import torch

from pinecone_setup import PineconeObject

from openai_api_call import openai_api

# Constant
NER_MODEL_PATH = "../../orkg-agriculture-ner/notebooks/orkgnlp-agriculture-ner"
NER_TOKENIZER = 'bert-base-cased'

class AgricultureNER:
    '''
    Apply the ORKG Agriculture NER developed by the Open Research Knowledge Graph.
    README of model: https://gitlab.com/TIBHannover/orkg/nlp/experiments/orkg-agriculture-ner/-/raw/v0.1.0/README.md
    Relevant Paper: https://doi.org/10.3390/knowledge4010001
    '''
    def __init__(self, model_path = None, tokenizer_path = None) -> None:
        '''
        Initiate the AgricultureNER class
        
        Parameter:
            model_path : the location of the ORKG Agriculture NER model
            tokenizer_path : the location of the ORKG Agriculture NER tokenizer model. 
                             It will be 'bert-base-cased' since it is a BERT based model. 
        '''
        if model_path is None:
            self.model_path = NER_MODEL_PATH
        else: self.model_path = model_path

        if tokenizer_path is None:
            self.tokenizer_path = NER_TOKENIZER
        else: self.tokenizer_path = tokenizer_path

    def inference(self, sentences: list) -> pd.DataFrame:
        '''
        inference applies (infers) the ORKG Agriculture NER model on the set of text.
        It returns the result in a pandas dataframe.

        Parameter:
            sentences (list): List of sentences. It can be the list of sentences identified by sentence identification (which uses spaCy) 
                               Sentence should be considered equivalent to the unit of analysis in other functions.

        Return:
            result_df_agrner (DataFrame): a dataframe of which each row represents a token, a entity tag, and a sentence that the toke comes from  
        '''
        result_list = []
        for sentence in sentences:

            # Ensure that the sentence is in a string format
            sentence = str(sentence)

            #Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            inputs = tokenizer(sentence, return_tensors="pt")

            # Import model
            model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            with torch.no_grad():
                logits = model(**inputs).logits

            predictions = torch.argmax(logits, dim=2)
            predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
            
            # Create a dictionary that has token, the entity class, and the original sentence (text divided by the unit of analysis)
            token_and_class = {i:{"token": inputs.tokens()[i], "entity": predicted_token_class[i], "sentence": sentence} for i in range(len(predicted_token_class)) if (predicted_token_class[i] == 'B-LOC') | (predicted_token_class[i] == 'I-LOC')}
            result_list.append(pd.DataFrame(token_and_class).transpose())

            # Convert the dictionary into a dataframe
            if result_list == []:
                self.result_df_agrner = pd.DataFrame(columns=["token", "entity", "sentence"])
            elif len(result_list) == 1:
                self.result_df_agrner = result_list[0]
            else:
                self.result_df_agrner = pd.concat(result_list, ignore_index= True)


        return self.result_df_agrner
    
    def entity_identification(self, agrner_result = pd.DataFrame(columns=["token", "entity", "sentence"]), beginning_token = "B-LOC") -> pd.DataFrame:
        '''
        entity_identification processes the previous result from the inference to identify a single entity refer by a number of tokens.
        Assume that there is a beginning tag for the token and includes all token until the next beginning tag

        Parameter:
            agrner_result (DataFrame): a dataframe that has token, entity, and sentence as the columns. 
                                       Assume that it is produced from the AgricultureNER.inference()

            beginning_token (str) : a string that identifies that it is the first token to represent an entity. (e.g. B-LOC for location)
                                    Check https://huggingface.co/learn/nlp-course/chapter7/2 for further rules in entity tagging.
        
        Return:
            entity_df_agrner(DataFrame): a dataframe that has 'token' and 'source' as a column.

        '''

        # Exception when no result is returned from ORKG Agriculture NER model.
        if agrner_result.shape[0] == 0:
            agrner_result = self.result_df_agrner

        if agrner_result.shape[0] == 0:
            self.entity_df_agrner = pd.DataFrame(columns= ["token", "source"])
            return self.entity_df_agrner
        
        else:
            # Calculate the indexes where the token begins
            agrner_result = agrner_result.reset_index()
            index_address = agrner_result[agrner_result.entity == beginning_token].index

            if sum(agrner_result.entity == beginning_token) == 0:
                index_address = agrner_result.head(1).index
            
            token_df_agg = []

            for i in range(len(index_address)):
                start = index_address[i]
                try: 
                    end = index_address[i+1]
                except:
                    end = agrner_result.tail(1).index.item()

                # Save the sentence that the token comes from
                sentence = agrner_result["sentence"].iloc[start]

                # Aggregate the tokens from a same entity to create a word 
                token_of_an_entity = [token for token in agrner_result["token"].iloc[start:end]] 
                token_of_an_entity = " ".join(token_of_an_entity)
                token_of_an_entity = token_of_an_entity.replace(" ##", "")

                # Combine the reconstructed entity and the sentence
                token_df = pd.DataFrame([token_of_an_entity, sentence]).transpose()
                token_df.columns = ["token", "source"]

                token_df_agg.append(token_df)

                self.entity_df_agrner = pd.concat(token_df_agg, ignore_index= True).reset_index()

            return self.entity_df_agrner

class RAGExtraction:
    def __init__(self, vector_storage: PineconeObject) -> None:
        '''
        Initiate a RAGExtraction object for the RAG pipeline
        
        Parameter:
            vector_storage (PineconeObject): a Pinecone object that will direct to the vector storage that has the document text as vectors.
        '''
        self.vector_storage = vector_storage

    def retrieve_relevant_document(self, query: str, document_title: str, text:list, include_score = True) -> list:
        '''
        Retrieve all vectors from the document based on a given query

        Parameter:
            query (str): a text that will be the basis of the cosine similarity measurement. All vectors related to the document will be returned, 
                         but the order will be based on how similar the text is to the query.
            document_title (str) : the title fo the document. It will be used to identify the namespace
            text (list) : the list that contains the string of the original text divided by the unit of analysis. 
            include_score (bool) : if true, it returns the cosine similarity score as well as the text.

        Return
            if include_score is true
                list : the first item will be the text returned and the next item will be the cosine similarity score between the query and the text. It should be the same length as the list text. 
            if include_score is false
                relevant_text (list): the text divided by the unit of analysis in a list. It will include all text from the document, but it will be in the order similar to the query.   
        '''
        # Embed the query
        query_embedding = self.vector_storage.embedding_text(query)
        
        # Retrieve vectors
        retrieved_info= self.vector_storage.index.query(vector=query_embedding, 
                                                        top_k=len(text), 
                                                        namespace = document_title,
                                                        include_metadata=True)
        
        # Extract the text from the vectors' metadata
        self.relevant_text = [match.metadata['text'] for match in retrieved_info.matches]
        # Extract the cosine simliarity scores
        self.cosine_value = [match['score'] for match in retrieved_info.matches]

        # Return result
        if include_score:
            return list(zip(self.relevant_text, self.cosine_value))
            
        else:
            return self.relevant_text
    
    def ask_rag_query(self, relevant_text = None, full_query_prompt = None) -> list:
        '''
        Generate a prompt that will integrate the text and the task and send it to OpenAI API

        Parameter:
            relevant_text (list) : the list of text that will be integrated into the prompt. If not given, it will uses the relevant_text variable of the object.
            full_query_prompt (str): a prompt that describe the task to the LLM. It will be considered as a "user" role. Text will be automatically added. If not specified, the default query will be implemented.
        '''
        self.openai_result = []

        if relevant_text is None:
            relevant_text = self.relevant_text

        # Integrate all relevant text to the prompt
        for text in relevant_text:
            
            if full_query_prompt is None:
                full_query = "Identify the geographical location in the text and return the relevant activity. Give a confidence score for each location and activity match when 0 is the lowest and 1 is the highest confidence.\
                Text: "
            else:
                full_query = full_query_prompt        
            
            # Construct the prompt
            full_query_dictionary = [
                {"role": "system", "content": "You are a context-catching machine that can only produce an output in this format:\n\n Location: [description of the location]\n Activity: [description of the activity]\n Confidence: [confidence rate] \n\n You can never use ':' except for formatting.\
                 When you cannot find an information, you can only answer 'Not Mentioned'. If there is not specific mention about the location, answer 'Not mentioned' in all categories.  You cannot keep any category empty. Also, you must describe the activity as detailed and accurately as possible based on the text. It needs to be a sentence with a subject and a verb,\
                 and it should clearly identify the crop variety. Exclude any information about the authors or the publisher of the document. "},
                {"role":"user", "content": full_query + text}
            ]

            # Send the prompt to the OpenAI API
            result = openai_api(full_query_dictionary)

            # Save result for the OpenAI call           
            self.openai_result.append(result) 
        
        # Extract only the content of the OpenAI call
        self.geographical_entity_response = [a_result.choices[0].message.content for a_result in self.openai_result]
        return self.geographical_entity_response, relevant_text
            
    def clean_rag_result(self, geographical_entity_response = None, relevant_text = None) -> pd.DataFrame:
        '''
        Convert the response from the OpenAI to a dataframe and clean the response (i.e. remove duplicates)
        Parameter:
            geographical_entity_response (list): the list of response from the OpenAI API after calling the ask_rag_query() function. If none, it will use the geographical_entity_response from the object
            relevant_text (list): the list of text that will be integrated into the prompt. It should be the same text used when ask_rag_query() function was called to create the geographical_entity_response.
                                  If none, it will use the relevant_text from the object
        
        Return:
            location_df (pd.DataFrame): a DataFrame with "Location", "Activity", "Confidence", and "Text" as columns
        '''
        if geographical_entity_response is None:
            geographical_entity_response = self.geographical_entity_response
        
        if relevant_text is None:
            relevant_text = self.relevant_text
        
        temporary_dict = {}
        df_collection = []

        ## Converting dataframe
        for i in range(0, len(geographical_entity_response)):
            # Process each response from the OpenAPI
            data_point = geographical_entity_response[i]
            # Recording the text that was included in the prompt for this OpenAI response
            relevant_text_of_data_point = relevant_text[i]

            # Split two when more than one location/activity was identified in a single OpenAI response
            identified_point = re.split("\n\n", data_point)
            
            for point in identified_point:
                # Divide by each feature. It should have Location, Activity, and Confidence
                feature = re.split("\n", point)

                # Convert the feature & value pairs into a dictionary
                for f in feature:
                    finding = re.search("^(.*):\s(.*)", f)
                    category = finding.group(1)
                    value = finding.group(2)
                    temporary_dict[category] = value 
                
                # Convert the feature & value dictionary into a dataframe
                temp_df = pd.DataFrame([temporary_dict])
                    
                # Impute missing values
                if "Location" not in temp_df.columns:
                    temp_df["Location"] = "Not mentioned"
                    
                if "Activity" not in temp_df.columns:
                    temp_df["Activity"] = "Not mentioned"

                if "Confidence" not in temp_df.columns:
                    temp_df["Confidence"] = np.nan

                # Convert missing value in Confidence as np.nan
                temp_df["Confidence"] = temp_df["Confidence"].replace("Not mentioned", np.nan)    
                
                # Save the originating text used for the prompt in the text column
                temp_df["Text"] = relevant_text_of_data_point

                # Append dataframe that represents each API response
                df_collection.append(temp_df) 

        # Aggregate each dataframe that represents a single API call into one dataframe.
        self.location_df_with_repetition = pd.concat(df_collection).reset_index(drop = True)

        ## Removing Duplicates
        dropping_row_index = []

        for row in range(0, self.location_df_with_repetition.shape[0]):
            # Set the row as a comparison point
            base_location = self.location_df_with_repetition.iloc[row].Location
            base_activity = self.location_df_with_repetition.iloc[row].Activity
            base_confidence = self.location_df_with_repetition.iloc[row].Confidence

            for comparing_row in range(0, self.location_df_with_repetition.shape[0]):
            # Compare each row with the comparison row (The if statement prevents the row from comparing with itself)   
                
                if row!=comparing_row:
                    # Get the value of the comparing row
                    comparing_location = self.location_df_with_repetition.iloc[comparing_row].Location
                    comparing_activity = self.location_df_with_repetition.iloc[comparing_row].Activity
                    comparing_confidence = self.location_df_with_repetition.iloc[comparing_row].Confidence

                    # If the location and the activity is identical
                    if (base_location == comparing_location) & (base_activity == comparing_activity):
                        base_confidence = float(base_confidence)
                        comparing_confidence = float(comparing_confidence)
                    # And if the base row has a lower comparison, the base row will be dropped.
                        if (base_confidence <= comparing_confidence):
                                dropping_row_index.append(row)

        # Collect all index of the rows that will be dropped
        dropping_row_index = list(set(dropping_row_index))

        # Drop the indexes based on the dropping_row_index
        self.location_df = self.location_df_with_repetition.drop(dropping_row_index)

        # Drop all location and activity that says "not mentioned"
        self.location_df = self.location_df[self.location_df.Location.str.lower() != "not mentioned"]
        self.location_df = self.location_df[self.location_df.Activity.str.lower() != "not mentioned"]
        
        # Reset the index
        self.location_df = self.location_df.reset_index(drop = True)

        return self.location_df




                