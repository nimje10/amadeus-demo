import os
from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
from openai import OpenAI
# Import Pinecone correctly if it's updated
from pinecone import Pinecone

QUERY_REFINER_OPENAI_MODEL = "gpt-4-turbo"
#QUERY_REFINER_OPENAI_MODEL = "gpt-3.5-turbo"

client = OpenAI(
    # This is the default and can be omitted
    #api_key=os.environ.get("OPENAI_API_KEY"),
    api_key=st.secrets["OPENAI_API_KEY"],
)

def find_match(input_text):

    # Use the OpenAI embeddings model
    try:
        # Generate embeddings
        embeddings = client.embeddings.create(input = [input_text], model="text-embedding-3-small").data[0].embedding
        #embeddings = response["embedding"]

        # Set up Pinecone
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "amadeus-chat"
        index = pc.Index(name=index_name)

        # Query Pinecone index with embeddings
        # result = index.query(embeddings, top_k=2)
        result = index.query(vector=[embeddings], top_k=5, include_metadata=True)
        print(result)
        # for match in result['matches']:
        #     print(f"{match['score']:.2f}: {match['metadata']['text']}")
        # return result
        if len(result['matches']) > 1:
            return "\n".join([match['metadata']['text'] for match in result['matches']])
        else:
            return "No sufficient matches found."
    except Exception as e:
        return f"Error generating embeddings or querying index: {str(e)}"


# def find_match(input_text):
#     # Securely load OpenAI API key from environment variables
#     openai.api_key = os.getenv('OPENAI_API_KEY')

#     # Replace the SentenceTransformer with OpenAIEmbeddings
#     # model = SentenceTransformer('all-MiniLM-L6-v2') # Old line
#     model = OpenAIEmbeddings(model_kwargs={'model_name': "ada"})
#     # Setting up Pinecone
#     PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  # It's also good to keep API keys out of the code
#     pc = Pinecone(api_key=PINECONE_API_KEY)
#     index_name = "amadeus-chat"
#     index = pc.Index(name=index_name)
#     input_em = model.encode(input_text).tolist()  # Ensure encode is the correct method if using OpenAIEmbeddings
#     result = index.query(input_em, top_k=2)
#     if len(result['matches']) > 1:
#         return "\n".join([match['metadata']['text'] for match in result['matches']])
#     else:
#         return "No sufficient matches found."

# def query_refiner(conversation, query):
#     # Updated API usage as per the latest documentation
#     try:
#         response = openai.Completion.create(
#             model="text-davinci-003",
#             prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#             temperature=0.7,
#             max_tokens=256,
#             top_p=1.0,
#             frequency_penalty=0,
#             presence_penalty=0
#         )
#         return response.choices[0].text.strip()
#     except Exception as e:
#         return f"Error in refining query: {str(e)}"
    
def query_refiner(conversation, query):
    try:
        model = QUERY_REFINER_OPENAI_MODEL #"gpt-3.5-turbo-1106"
        messages = [
                {"role": "system", "content": "You are an expert prompt engineer in querying documents having multiple rule based information"},
                {"role": "user", "content": f"Given the following user query and conversation log, formulate a question, without missing out any key information, that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"},
            ]
        response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0
            )
        response_message = response.choices[0].message.content
        print("refined query", response_message)
        return response_message
    except Exception as e:
        return f"Error in refining query: {str(e)}"


def get_conversation_string():
    conversation_string = ""
    responses = st.session_state.get('responses', [])
    requests = st.session_state.get('requests', [])
    for i in range(len(responses)-1):
        conversation_string += f"Human: {requests[i]}\n"
        conversation_string += f"Bot: {responses[i+1]}\n"
    return conversation_string

# Always check for required session state keys
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
