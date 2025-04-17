# Speech-to-Text Voicebot

A RAG-based voice assistant that answers questions about a provided PDF—using only your voice.

---

## Overview

**speech-to-text-voicebot** is an AI-powered voice assistant designed to answer your questions about a specific PDF document. The system uses speech recognition to take your voice as input, processes your query, and responds with answers strictly based on the content of the PDF provided in the repository. This project demonstrates a Retrieval-Augmented Generation (RAG) workflow, combining speech-to-text technology with document-based question answering.

---

## Features

Voice-Only Interaction: Ask questions using your voice; no typing required.  
PDF-Based Answers: The bot answers strictly based on the content of the provided PDF.  
Embeddings Workflow: Efficient retrieval of relevant information using document embeddings.  
Modular Codebase: Core logic separated into `embed.py`, `final.py`, and `utils.py` for clarity and maintainability.

---

## Project Workflow

1. Generate PDF Embeddings  
   Run `embed.py` to process the provided PDF and store document embeddings for efficient retrieval.

2. Start the Voicebot  
   Run `final.py` to launch the voice assistant.  
   `final.py` imports utility functions from `utils.py` to handle speech recognition, query processing, and answer generation.

3. Ask Questions  
   Interact with the bot using your voice. The bot transcribes your question, searches the PDF for relevant information, and responds accordingly.

---

## Getting Started

### Prerequisites

Python 3.12  
Required Python libraries (see `requirements.txt`)  
Microphone for voice input  
OpenAI and Pinecone client API keys

### Usage

1. Clone the repository:

    `git clone https://github.com/Jugz-08/speech-to-text-voicebot.git`

    `cd speech-to-text-voicebot`


2. Install dependencies:

   `pip install -r requirements.txt`


3. Embed the PDF:  
Place your PDF in the repository folder (or use the sample provided).  
Run:  

    `python embed.py`


4. Start the voicebot:  

    `python final.py`


5. Interact:  
Use your microphone to ask questions related to the PDF.  
The bot will respond based on the document content.

---

## How It Works

| Step                | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| PDF Embedding       | `embed.py` processes the PDF and stores vector embeddings for retrieval.     |
| Voice Input         | `final.py` captures your spoken question using speech-to-text technology.    |
| Query Processing    | The system searches the embedded PDF for relevant content.                   |
| Response Generation | The bot formulates an answer using only information from the PDF.            |

---

## Example Use Case

User: (Speaks) "What is the main topic of the document?"  
Bot: (Responds) "The main topic of the document is ..."

---
