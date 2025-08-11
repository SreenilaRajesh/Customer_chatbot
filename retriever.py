from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
import config
import qdrant_operations

class Retriever:
    def __init__(self):
        self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.embedding_model = LateInteractionTextEmbedding(model_name=config.EMBEDDING_MODEL, cache_dir=config.EMBEDDING_MODEL_PATH)

    def retrive_chunks(self, collection_name, query, k):
        query = query.lower()
        query_embedding = list(self.embedding_model.embed(query))[0]
        result = qdrant_operations.get_querypoints_in_collection(
            qdrant_client=self.client,
            collection_name=collection_name,
            query=query_embedding,
            k=k
        )
        #print(result)
        retrieved_chunks = [point.payload['text'] for point in result.points]
        return retrieved_chunks
    

#if __name__ == "__main__":
#    retriever = Retriever()
#    print(retriever.retrive_chunks(config.COLLECTION_NAME, "What is the purpose of the Autodesk Chatbot?", 5))
    

