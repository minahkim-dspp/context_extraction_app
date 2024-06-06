# Import Packages and Modules
from text_preprocessing import pdfreader_text_extraction, text_cleaning, sentence_division, address_identification, text_splitter
from entity_extraction import AgricultureNER, RAGExtraction
from pinecone_setup import PineconeObject
import re

class Document:
    def __init__(self, path, chunk_size = 1024, chunk_overlap = 100, index_name = "chunked-document"):
        self.path = path
        self.name = re.search(r"([^/]+)\.pdf$", path).group(1)
        self.text = pdfreader_text_extraction(self.path)
        self.text = text_cleaning(self.text)
        self.sentences = sentence_division(self.text)
        self.chunk = text_splitter(self.text, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        self.vector_storage = PineconeObject(index_name)
    
    def geographical_extraction(self, method = 'rag', unit_of_analysis = "chunk" , **kwargs):
        '''
        Extract the Geographical Extraction from the document

        Parameter:
            method(str): the method used to extract entity relevant to geographical region. 
            The two options are 'extraction_chain' and "agriculture-ner"
            unit_of_analysis (str) : the level of analysis. If the unit_of_analysis is "chunk", geographical extraction will happen in a chunk level.
                                     If the unit of analysis is "sentence", geographical extraction will happen in a sentence level.

        '''

        if self.text == "":
             return self

        if unit_of_analysis == "chunk":
            unit_of_analysis = self.chunk
        elif unit_of_analysis == "sentence":
            unit_of_analysis = self.sentences


        ## Using ORKG Agriculture NER Model 
        if method == "agriculture-ner":
            # Set parameters for models
            if "model_path" in kwargs:
                model_path = kwargs.model_path
            else: model_path = None

            if "tokenizer_path" in kwargs:
                tokenizer_path = kwargs.tokenizer_path
            else: tokenizer_path = None

            # NER inference using Agriculture NER model
            self.agrculture_ner = AgricultureNER(model_path, tokenizer_path)
            self.geographical_result = self.agrculture_ner.inference(sentences = unit_of_analysis)
            try:
                # Clean the entity
                self.geographical_result = self.agrculture_ner.entity_identification(agrner_result = self.geographical_result)
                
                # Take away address in the result 
                boolean_address =[address_identification(text = text) == False for text in self.geographical_result.source]
                self.geographical_result = self.geographical_result[boolean_address]
                
            except:
                pass

        ## Using RAG
        if method == "rag":
            # Set up a vector storage by converting text into vectors and upserting them. 
            self.vector_storage.vector_storage_setup(text= unit_of_analysis, title = self.name, path = self.path)

            # Set up a RAG Extraction Object (connecting the vector storage that has the text vectors to the RAG pipeline)
            self.rag_object= RAGExtraction(vector_storage = self.vector_storage)

            # Retrieve vectors
            self.rag_object.retrieve_relevant_document(query = "This practice happens in this geographical region.", document_title=self.name, text= unit_of_analysis)
            
            # If the vector storage does not return anything, try once again. If not, then raise an error
            if len(self.rag_object.relevant_text) ==0 :
                self.rag_object.retrieve_relevant_document()
                if len(self.rag_object.relevant_text) ==0:
                    raise ValueError ("No Vector returned from storage")

            # Ask query
            self.rag_object.ask_rag_query()

            # Clean the result
            self.rag_result = self.rag_object.clean_rag_result()
            
            return self.rag_result