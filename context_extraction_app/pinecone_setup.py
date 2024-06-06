import os
import time
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
API_KEY = os.environ.get('PINECONE_API_KEY')

class PineconeObject:
    def __init__(self,
                 index_name: str, 
                 embedding_model = EMBEDDING_MODEL,
                 delete_index = None, 
                 index_dimension = 1024, 
                 metric = "cosine",
                 serverlessspec_cloud = 'aws',
                 serverlessspec_region = "us-east-1") -> None:
        # Initialize pinecone client
        self.pinecone_client = Pinecone(api_key = API_KEY)
        try:
            self.index = self.pinecone_client.Index(index_name) 
        except:
            self.index = self.set_index(index_name, delete_index, index_dimension, metric, serverlessspec_cloud, serverlessspec_region)

        # Set Embedding Model
        self.embedding_model = SentenceTransformer(embedding_model)


    def set_index(self, index_name: str, 
                    delete_index = None, 
                    index_dimension = 1024, 
                    metric = "cosine",
                    serverlessspec_cloud = 'aws',
                    serverlessspec_region = "us-east-1") -> Pinecone.Index:

            # Delete Previously Existing Index before moving on
            if delete_index is not None:
                if delete_index == True:
                    delete_index = index_name
                else:
                    delete_index = str(delete_index)
                self.delete_index(delete_index)
                
            # Create Index
            self.pinecone_client.create_index(
                name = index_name,
                dimension = index_dimension,
                metric = metric,
                spec = ServerlessSpec(
                    cloud = serverlessspec_cloud,
                    region = serverlessspec_region
                )
            )

            # Set up Index
            index = self.pinecone_client.Index(index_name)
            print(index.describe_index_stats())

            return index
    
    def delete_index(self, delete_index: str) -> None:
        '''
         Delete Index fron the Pinecone Database
        '''
        try:
            self.pinecone_client.delete_index(delete_index)
        except:
             print(f"There was an error while deleting the index {delete_index}")


    def embedding_text(self, text) -> list:
        result = self.embedding_model.encode(text).astype(float)
        result = list(result)
        return result

    def vector_storage_setup(self, text: list, title: str, no_replica = True, embedding_dimension = 1024, **metadata_kwargs):
        '''
        Set up the vector storage to include the document's text 
        Parameter:
            text (list): the text divided by the unit of analysis. These text will be converted into embeddings and upserted to the vector storage.
            title (str): the title of the document. It will identify the namespace.
            no_replica (bool): if true, all existing vectors in the namespace that is same as the title will be erased before the method upsert the new vectors.
            embedding_dimension (int): the dimension of the arrays when the embedding model convert a text into a vector
            metadata_kwargs (dict): these values will be included into the metadata when vectors are upserted
        
        Return:
            self.index.describe_index_stats() : describe the status of the index
        '''
        # Batch size when upserting or deleting the vectors. It will be 100 if the number of vectors in the namespace is larger than 100, and 10 when it is smaller 
        if len(text) > 100:
            batch_size = 100
        else: batch_size = 10

        # Check if the same vector exists
        index_stat = self.index.describe_index_stats()
        if title in index_stat.namespaces.keys():
            # Removing replica when no_replica is true
            if no_replica:
                query_vector = np.random.uniform(-1, 1, size=embedding_dimension).tolist()
                query_num = index_stat.namespaces[title]["vector_count"]
                replicas = self.index.query(vector = query_vector, top_k = query_num, namespace=title)
                ids_total = [replica["id"] for replica in replicas["matches"]]

                # Raise error when the query does not return the vectors
                if len(ids_total) != query_num:
                    raise ValueError("There needs to be ", query_num, " IDs but it only returned ", len(ids_total), "  IDs")

                # Delete indexes in batches (10 or 100)
                for i in range(0, query_num, batch_size):
                    i_end = min(query_num, i+batch_size)
                    ids = ids_total[i:i_end]

                    if len(ids) == 0:
                        raise ValueError("No ID is returned from the query between ", i, " and ", i_end, " and replicas return ", replicas)

                    self.index.delete(ids = ids, namespace = title)
                    time.sleep(0.01)

        for i in range(0, len(text), batch_size):
            # find end of batch
            i_end = min(len(text), i + batch_size)

            # List to store vectors for upsertion
            to_upsert = []  

            for n in range(i, i_end):
                # Set an ID
                id = str(n)

                # Embed the text in the batch
                values = self.embedding_text(text[n])  
                # Ensure that the values are in float
                values = [float(value) for value in values]

                # Include the text and the metadata_kwargs as a metadata
                metadata =  {
                    "text": text[n]
                }
                metadata.update(metadata_kwargs)
                
                # Create dictionary representing the vector
                vector_dict = {'id': id, 'values': values, 'metadata': metadata}  

                # Append dictionary to list
                to_upsert.append(vector_dict)  

            # Upsert the list of vectors  
            self.index.upsert(to_upsert, namespace = title)                       
            
        return self.index.describe_index_stats()