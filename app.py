from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
# from retrieve_result import retreival_result, result_after_retreival
from retrieve_result import retrieval_result, result_after_retrieval


from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings  # Correct updated import

from dotenv import load_dotenv
import os

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Create Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Set up embeddings
embeddings = download_hugging_face_embeddings()

# Set up LangChain Pinecone VectorStore (serverless style)
index_name = "medicalbot"

docsearch = PineconeVectorStore(
    index_name=index_name,
    namespace="",  # optional
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
    # pinecone_environment=PINECONE_ENVIRONMENT
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg.lower()
    print(input_text)
    docs = retrieval_result(PINECONE_API_KEY, input_text, docsearch)

    # docs = retreival_result(PINECONE_API_KEY, input_text, docsearch)

    # Process the documents and generate a response
    
    response = result_after_retrieval(GROQ_API_KEY, input_text, docs)

    # response = result_after_retreival(GROQ_API_KEY, input_text, docs)

    # Concatenate the response into a single string
    full_response = ''.join(response)
    print(full_response)

    # Handle general responses based on content
    if any(greeting in input_text for greeting in ["hi", "hello", "hey"]):
        return "Hello! How can I assist you today?"
    elif any(farewell in input_text for farewell in ["bye", "goodbye"]):
        return "Goodbye! Take care."
    elif "thanks" in input_text or "thank you" in input_text:
        return "You're welcome! Let me know if you have any other questions."
    elif full_response:  # If there is a relevant response
        return full_response
    else:  # Fallback response for unclear queries
        return "I'm sorry, I'm not sure about that."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081, debug=True)










# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from retrieve_result import retreival_result, result_after_retreival

# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings

# from langchain.vectorstores import Pinecone as LangchainPinecone  # Using alias for LangChain Pinecone

# from dotenv import load_dotenv
# import os



# app = Flask(__name__)

# load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"] = GROQ_API_KEY

# embeddings = download_hugging_face_embeddings()

# index_name = "medicalbot"

# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = LangchainPinecone.from_existing_index(index_name, embeddings)

# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = str.lower(msg)
#     print(input)
#     docs = retreival_result(PINECONE_API_KEY,input,docsearch)
#     # Process the documents and generate a response
#     response = result_after_retreival(GROQ_API_KEY,input,docs)
#     # Concatenate the response into a single string
#     full_response = ''.join(response)
#     print(full_response)
#     # Handle general responses based on content
#     if any(greeting in input for greeting in ["hi", "hello", "hey"]):
#         return "Hello! How can I assist you today?"
#     elif any(farewell in input for farewell in ["bye", "goodbye"]):
#         return "Goodbye! Take care."
#     elif "thanks" in input or "thank you" in input:
#         return "You're welcome! Let me know if you have any other questions."
#     elif full_response:  # If there is a relevant response
#         return full_response
#     else:  # Fallback response for unclear queries
#         return "I'm sorry, I'm not sure about that."




# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 8081, debug= True)
