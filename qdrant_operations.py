from qdrant_client import models
import uuid

def is_collection_available(qdrant_client, collection_name):
    return qdrant_client.collection_exists(collection_name=collection_name)

def setup_collection(qdrant_client, collection_name):
    """
    Set up the Qdrant collection with the specified vector configurations.
    Args:
        qdrant_client (QdrantClient): The Qdrant client instance.
        collection_name : Name of the collection to be created
    """
    vector_config = models.VectorParams(
        size=128,  # size of each vector produced by ColBERT
        distance=models.Distance.COSINE,  # similarity metric between each vector
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        )
    )
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=vector_config,
        on_disk_payload=True
    )

def upload_points_to_collection(qdrant_client, collection_name, embeddings, metadata):
    qdrant_client.upload_points(
        collection_name = collection_name,
        points = [
            models.PointStruct(
                id = str(uuid.uuid4()),
                vector = vector,
                payload = metadata[idx]
            )
            for idx, vector in enumerate(embeddings)
        ],
        max_retries = 3,
    )


def get_querypoints_in_collection(qdrant_client, collection_name, query, k):
    result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query,
            limit=k,
            with_payload=True
        )

    return result













