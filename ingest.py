
import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import qdrant_client
from qdrant_client.models import VectorParams, Distance, PointStruct

# Optional: set custom cache directory
os.environ["HF_HOME"] = r"C:\Users\visah\Documents\GitHub\Autodesk_Chatbot\cache_dir"

MD_FOLDER = "markdown_files_crawler"
COLLECTION_NAME = "autodesk_markdown_chunks"

client = qdrant_client.QdrantClient(host="localhost", port=6333)


def load_markdown_documents(folder_path):
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".md"):
                file_path = os.path.join(root, file)
                loader = UnstructuredMarkdownLoader(file_path)
                docs.extend(loader.load())
    return docs

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


documents = load_markdown_documents(MD_FOLDER)
chunks = chunk_documents(documents)


embedding_model = HuggingFaceEmbeddings(
    model_name="colbert-ir/colbertv2.0",
    model_kwargs={"device": "cpu"}
)

doc_embeddings = embedding_model.embed_documents(chunks)

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=128,
        distance=Distance.MultiVectorComparator.MAX_SIM
    )
)

client.upload_points(
    collection_name=COLLECTION_NAME,
    points=[
        PointStruct(
            id=i,
            vector=embedding
        )
        for i, embedding in enumerate(doc_embeddings)
    ]
)

print(f"Indexed {len(chunks)} chunks into local Qdrant.")
