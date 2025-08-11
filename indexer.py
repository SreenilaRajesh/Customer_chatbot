from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
from chunker import load_markdown_documents, chunk_documents
import config
import qdrant_operations

class Indexer:
    def __init__(self):
        self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.embedding_model = LateInteractionTextEmbedding(model_name=config.EMBEDDING_MODEL, cache_dir=config.EMBEDDING_MODEL_PATH)
        self.batch_size = config.EMBEDDING_BATCH_SIZE

    def __index_chunks(self, chunks, collection_name):
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks_embeddings_list = list(
                self.embedding_model.embed(chunks[i:i + self.batch_size])
            )
            chunks_list = [{'text': chunk} for chunk in chunks[i:i + self.batch_size]]
            qdrant_operations.upload_points_to_collection(
                qdrant_client=self.client,
                collection_name=config.COLLECTION_NAME,
                embeddings=batch_chunks_embeddings_list,
                metadata=chunks_list
            )

    def index_files(self):
        documents = load_markdown_documents(config.MD_FOLDER)
        chunks = chunk_documents(documents)
        all_texts = [c.page_content if hasattr(c, "page_content") else str(c) for c in chunks]
        if not (qdrant_operations.is_collection_available(self.client, config.COLLECTION_NAME)):
            qdrant_operations.setup_collection(self.client, collection_name=config.COLLECTION_NAME)

        self.__index_chunks(all_texts, config.COLLECTION_NAME)


#if __name__ == "__main__":
#    indexer = Indexer()
#    indexer.index_files()







