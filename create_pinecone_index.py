import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

print("Loaded Pinecone API Key:", os.getenv('PINECONE_API_KEY'))
print("Loaded Pinecone Environment:", os.getenv('PINECONE_ENVIRONMENT'))

# Create a Pinecone client
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Check if index already exists (optional good practice)
index_name = "medicalbot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",  # best for sentence-transformers
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Serverless index '{index_name}' created successfully!")
else:
    print(f"Index '{index_name}' already exists, skipping creation.")
