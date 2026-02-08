# PharmaAssist-LLM: Pharmaceutical Assistant (PoC) 🏥💊

This repository contains a **Proof of Concept (PoC)** for an AI-powered Pharmaceutical Assistant. The project integrates advanced Natural Language Processing (NLP) techniques, including **Fine-Tuning** and **RAG (Retrieval-Augmented Generation)** (RAFT), to provide accurate information about medications, drug compositions and pharmacological mechanisms.

## Key Features
* **Specialized Data Pipeline:** Advanced cleaning and normalization of pharmaceutical datasets (medicine names, chemical compositions and manufacturers).
* **Efficient Fine-Tuning:** Leveraging `Unsloth` for 2x faster and memory-optimized training of Large Language Models (LLMs) like Llama-3.
* **Hybrid RAG/RAFT Architecture:** Combines a specialized vector database (`FAISS`) with medical context (Kaggle datasets + Katzung Basic & Clinical Pharmacology) to minimize hallucinations.
* **Safety-First Benchmarking:** Evaluation framework focused on context fidelity, medical accuracy, and safety guardrails.

## Project Structure
The workflow is organized into five sequential stages:

1.  **`01_Data_Cleaning`**: Initial processing of raw pharmaceutical data and text normalization.
2.  **`02_Pharma_RAFT`**: Creation of the semantic knowledge base (FAISS Vector Store) using `HuggingFace` embeddings.
3.  **`03_FineTuning`**: Model optimization using QLoRA and Unsloth to adapt the LLM to the medical domain.
4.  **`04_Inference_RAG`**: The core assistant logic, combining the fine-tuned model with real-time document retrieval.
5.  **`05_Benchmarking`**: Comparative analysis of model outputs to ensure reliability and medical consistency.

## Results Analysis and Conclusions

A triple-blind evaluation was implemented to compare the performance of three different models:
Gemma-3-4B Base, Gemma-3-4B with a combination of a RAG system and QLoRA and Qwen 2.5 32B.
A Llama-3.3-70B model was utilized as an "AI Judge" (AI-as-a-Judge). It evaluated 3 critical metrics on a scale of 1 to 10 (fidelity, accuracy and safety), 
based on questions randomly generated from the local knowledge base (Katzung Textbook + Kaggle Dataset).



## Future Improvements

* To scale this project into a production-level tool, the following development lines are proposed:
* Agentic RAG (Multi-stage Agents): Implement reasoning logic where the model decides whether it needs to search the technical textbook (Katzung) or the commercial list (Kaggle) before formulating a response.
* Multimodal Support: Integrate the capability to analyze images of medical prescriptions or drug labels using Vision Language Models (VLM).
* Human Expert Evaluation: Conduct a comparative benchmark where licensed pharmacists validate the AI Judge's ratings to calibrate the sensitivity of the evaluation.
* Expansion of the Knowledge Base: Include Clinical Practice Guidelines (CPGs) and more extensive drug-drug interaction databases to cover complex polypharmacy cases.

## Stack

Python, Unsloth, Llama-3, Hugging Face Transformers, PEFT, 
LangChain, FAISS, Groq Cloud, Pandas, NumPy, PyPDF.

## Dataset

* https://www.kaggle.com/datasets/singhnavjot2062001/11000-medicine-details/data
* Katzung - Basic and Clinical Pharmacology 12th Edition (2012)
* General Principles of Pharmacology

## Prerequisites
* Python 3.10+
* NVIDIA GPU (T4 or better recommended for Unsloth)
* Required API Keys/Tokens:
    * **Groq API Key**: For accelerated cloud inference.
    * **Hugging Face Token**: To access and download base models.
