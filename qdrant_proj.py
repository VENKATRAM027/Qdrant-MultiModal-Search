import tensorflow as tf
from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams,PointStruct
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
from dotenv import load_dotenv
from transformers import AutoProcessor,TFCLIPModel

#Loading Qdrant Credentials

load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
    )

#Loading the model

model_id = "openai/clip-vit-base-patch-32"
model = TFCLIPModel.from_pretrained(model_id,from_tp=True)

#Initializing the database table
collection_name = "tf_multimodal_search"

if not client.collection_exists(collection_name):
    client.create_collection(collection_name=collection_name),
    vectors_config=VectorParams(size=512,distance=Distance.COSINE
)

#Data preprocessing & Tensor conversion

database_texts = [
    "A golden retriever playing in the park",
    "A steaming cup of dark roast coffee on a wooden table",
    "A sleek, red sports car driving down a coastal highway",
    "A futuristic city skyline illuminated by neon lights at night"
]