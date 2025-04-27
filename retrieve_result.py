from groq import Groq
from langchain.vectorstores import Pinecone as LangchainPinecone  # Using alias for LangChain Pinecone

def retrieval_result(api_key, query, docsearch):
    if not api_key:
        print("API Key not found. Please set the PINECONE_API_KEY environment variable.")
        return []
    else:
        docs = docsearch.similarity_search(query, k=3)
        return docs

def result_after_retrieval(api_key, query, docs):
    if not api_key:
        print("API Key not found. Please set the GROQ_API_KEY environment variable.")
        return "API Key missing."
    else:
        client = Groq(api_key=api_key)

        # Prepare the relevant information from the retrieved documents
        relevant_information = "\n".join(doc.page_content for doc in docs)

        # Prepare the relevance check input
        relevance_check_input = f"""
            Based on the following user query and relevant information, determine if the information is relevant to the query. 
            If it is relevant, provide a concise and clear answer addressing the user's question; if not, respond with "I'm sorry, I'm not sure about that."

            User Query: {query}

            Relevant Information:
            {relevant_information}
        """

        try:
            # Create the chat completion request with streaming enabled
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "user", "content": relevance_check_input}
                ],
                temperature=0.8,
                max_tokens=1024,
                top_p=0.65,
                stream=True,
                stop=None
            )

            # Buffer to collect the streamed response
            full_response = []

            # Handle the streamed chunks
            for chunk in completion:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_response.append(delta)

            return full_response  # Normal case ✅

        except Exception as e:
            print(f"An error occurred during the API call: {str(e)}")
            return [f"I'm sorry, I couldn't process your request. (Error: {str(e)})"]





# from groq import Groq
# from langchain.vectorstores import Pinecone as LangchainPinecone  # Using alias for LangChain Pinecone

# def retreival_result(api_key,query,docsearch):
#     if not api_key:
#         print("API Key not found. Please set the PINECONE_API_KEY environment variable.")
#     else:
#         docs = docsearch.similarity_search(query, k=3)
#         return docs

# def result_after_retreival (api_key, query, docs):
#     if not api_key:
#         print("API Key not found. Please set the GROQ_API_KEY environment variable.")
#     else:
#         client = Groq(api_key=api_key)

#         # Prepare the relevant information from the retrieved documents
#         relevant_information = "\n".join(doc.page_content for doc in docs)

#         # Prepare the relevance check input
#         relevance_check_input = f"""
#             Based on the following user query and relevant information, determine if the information is relevant to the query. 
#             If it is relevant, provide a concise and clear answer addressing the user's question; if not, respond with "I'm sorry, I'm not sure about that."

#             User Query: {query}

#             Relevant Information:
#             {relevant_information}
#             """

#         try:
#             # Create the chat completion request with streaming enabled
#             completion = client.chat.completions.create(
#                 model="llama3-8b-8192",
#                 messages=[
#                     {"role": "user", "content": relevance_check_input}
#                 ],
#                 temperature=0.8,
#                 max_tokens=1024,
#                 top_p=0.65,
#                 stream=True,  # Streaming enabled
#                 stop=None
#             )

#             # Buffer to collect the streamed response
#             full_response = []

#             # Handle the streamed chunks
#             for chunk in completion:
#                 # Access the delta content from the chunk
#                 if hasattr(chunk, 'choices') and chunk.choices:
#                     delta = chunk.choices[0].delta.content
#                     if delta:
#                         full_response.append(delta)  # Collect the response chunks

#             # Join and print the full response once all chunks are received
#             #print("Full Response: ", ''.join(full_response))
#             return full_response

#         except Exception as e:
#             print(f"An error occurred during the API call: {str(e)}")