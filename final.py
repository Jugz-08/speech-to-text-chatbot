import os
import streamlit as st
import base64
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import OpenAI as LangchainOpenAI  # Correct LLM class
from langchain.memory import ConversationBufferWindowMemory
from openai import OpenAI  # For audio transcription and TTS
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
from utils import text_to_speech, autoplay_audio, speech_to_text
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
api_key_openai = st.secrets['OPENAI_API_KEY']
api_key_pinecone = st.secrets['PINECONE_API_KEY']
index_name = "cvc"

# ----------------------------------------------------

# Set up Pinecone API
def setup_pinecone_api(api_key):
    if "PINECONE_API_KEY" not in os.environ:
        os.environ["PINECONE_API_KEY"] = api_key

# Create a retrieval chain for conversational AI
def create_retrieval_chain(vectorstore, memory):
    chain = ConversationalRetrievalChain.from_llm(
        LangchainOpenAI(),
        vectorstore.as_retriever(search_kwargs={'k': 3}),
        memory=memory,
        condense_question_prompt=condense_question_prompt_template,
        combine_docs_chain_kwargs=dict(prompt=qa_prompt),
        verbose=True
    )
    return chain

# Perform conversational retrieval
def perform_conversational_retrieval(chain, query):
    memory = st.session_state["context_mem"]
    output = chain({"question": query, "chat_history": memory.load_memory_variables({})["chat_history"]})
    return output["answer"]

# Initialize Pinecone vector store
def initialize_vectorstore():
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')  # Define embeddings to retrieve relevant documents
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace="")
    return vectorstore

# Prompt templates
_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a 
standalone question without changing its content. If the question is not related to the previous context, do not rephrase it; answer directly.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

condense_question_prompt_template = PromptTemplate.from_template(_template)

prompt_template = """You are an helpful medical assistant. Your job is to answer questions from the context given below.\
 Make sure you DO NOT ANSWER anything NOT related to the given context, in such cases just reply "Out of scope question"  
If you don't know the answer,\
 just say that you don't know. Don't try to make up an answer. If the question is not related to\
 previous context,Please DO NOT rephrase it. Answer the question directly.

{context}

Question: {question}
Helpful Answer:"""


qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

setup_pinecone_api(api_key_pinecone)

# Streamlit UI for voice input/output
def main():
    st.title("Conversational Chatbot with RAG 🤖")

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "context_mem" not in st.session_state:
        st.session_state["context_mem"] = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

    # Footer container for microphone input
    footer_container = st.container()
    with footer_container:
        audio_bytes = audio_recorder()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Process recorded audio if available
    if audio_bytes:
        with st.spinner("Transcribing..."):
            webm_file_path = "temp_audio.mp3"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)
            transcript = speech_to_text(webm_file_path)
            if transcript:
                st.session_state.messages.append({"role": "user", "content": transcript})
                with st.chat_message("user"):
                    st.write(transcript)
                os.remove(webm_file_path)

    # Generate response if last message is from user
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking 🤔..."):
                vectorstore = initialize_vectorstore()
                print(vectorstore)
                chain = create_retrieval_chain(vectorstore, st.session_state["context_mem"])
                print(chain)
                final_response = perform_conversational_retrieval(chain, st.session_state.messages[-1]["content"])
                print(final_response)
            
            with st.spinner("Generating audio response..."):
                audio_file = text_to_speech(final_response)
                autoplay_audio(audio_file)

            st.write(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            os.remove(audio_file)

if __name__ == "__main__":
    setup_pinecone_api(api_key_pinecone)
    main()