# RAG-Based Driving Instructor

## Overview

RAG-Based Driving Instructor is an innovative application that leverages Retrieval Augmented Generation (RAG) to provide instant, accurate answers to questions about driving in Ontario. Powered by NVIDIA NIM (NVIDIA Inference Microservices), this app serves as a virtual driving instructor, offering reliable information sourced directly from the Ontario Driver's Handbook.

## Features

- **Real-time Q&A**: Get immediate answers to your driving-related questions.
- **Official Information**: All responses are based on the official Ontario Driver's Handbook.
- **AI-Powered**: Utilizes state-of-the-art AI technology from NVIDIA for accurate and context-aware answers.
- **User-Friendly Interface**: Simple, intuitive Streamlit-based web interface.

## How It Works

1. The app uses a PDF of the Ontario Driver's Handbook as its knowledge base.
2. When you ask a question, the app searches through this knowledge base to find relevant information.
3. It then uses NVIDIA's advanced language model to generate a response based on the relevant information and your question.

## Getting Started

To use the RAG-Based Driving Instructor, simply visit our web application:

[RAG-Based Driving Instructor App](https://rag-driving-instructor.streamlit.app/)

Note: You'll need an NVIDIA API key to use the application. If you don't have one, you can obtain it from the [NVIDIA AI Playground](https://build.nvidia.com/explore/discover).

## Local Development

If you want to run the app locally or contribute to its development:

1. Clone this repository: `git clone https://github.com/Gaurav-612/rag-driving-instructor.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## Technologies Used

- Python
- Streamlit
- LangChain
- NVIDIA NIM
- FAISS for vector storage
