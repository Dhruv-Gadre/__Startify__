# Software Requirements Specification (SRS)

## Project Title
Startify: AI-Based Startup Idea Evaluation System

## 1. Introduction

### 1.1 Purpose
The purpose of this project is to develop a software system that helps users evaluate startup ideas using artificial intelligence and machine learning techniques. The system analyzes a user-provided startup idea, compares it with existing startups, predicts a likely industry, checks similarity with existing companies, and estimates possible funding potential.

This document defines the functional and non-functional requirements of the Startify system. It is intended for academic submission, project understanding, and future development.

### 1.2 Scope
Startify is a preliminary startup evaluation platform designed for students, founders, and early-stage innovators. The system accepts a startup description, city, and founding year as input. It then performs the following tasks:

1. Finds similar startups from an existing dataset.
2. Predicts the likely industry of the idea.
3. Uses a local AI model to perform a similarity or copycat analysis.
4. Shows funding-related insights from matched startups.
5. Predicts possible funding amount using a trained machine learning pipeline.

The project includes a static web interface, a Streamlit prototype interface, a FastAPI backend, a local Ollama-based AI analysis module, and a machine learning model for funding prediction.

### 1.3 Definitions, Acronyms, and Abbreviations
SRS: Software Requirements Specification

API: Application Programming Interface

UI: User Interface

ML: Machine Learning

LLM: Large Language Model

FAISS: Facebook AI Similarity Search

REST: Representational State Transfer

FastAPI: Python framework for building APIs

Streamlit: Python framework for creating quick data applications

Ollama: Local runtime for running LLM models


## 2. Specific Requirements

### 2.1 Functional Requirements

#### 2.1.1 User Input
1. The system shall accept a startup idea description from the user.
2. The system shall accept the user's city.
3. The system shall accept the planned founding year.
4. The system shall validate that all required inputs are provided before analysis starts.

#### 2.1.2 Similar Startup Search
1. The system shall convert startup descriptions into embeddings using a sentence transformer model.
2. The system shall create and use a FAISS index for similarity search.
3. The system shall retrieve the top matching startups from the dataset.
4. The system shall display the matched startup records to the user.

#### 2.1.3 Industry Prediction
1. The system shall determine the most likely industry based on the matched startup results.
2. The system shall show the predicted industry in the final output.

#### 2.1.4 AI Similarity Analysis
1. The system shall send the user idea and top matched startup description to the local LLM service.
2. The system shall receive a similarity score from 0 to 100.
3. The system shall receive a copycat verdict as true or false.
4. The system shall receive reasoning text explaining the verdict.
5. The system shall display the AI similarity analysis to the user.

#### 2.1.5 Funding Analysis
1. The system shall calculate how many of the top matched startups received funding.
2. The system shall calculate the average funding amount for funded matches.
3. The system shall display these funding statistics.

#### 2.1.6 Funding Prediction
1. The system shall load the saved funding prediction pipeline.
2. The system shall generate features from user input.
3. The system shall predict an estimated funding amount.
4. The system may provide a confidence value and prediction range.
5. The system shall display an error message if the funding model is not available.

#### 2.1.7 API Support
1. The backend shall provide an API endpoint for startup analysis.
2. The API shall accept JSON input containing idea, city, and founding year.
3. The API shall return structured JSON output containing profile details, AI analysis, similar startups, funding analysis, and funding prediction.

#### 2.1.8 Training Module
1. The system shall support offline model training using the startup dataset.
2. The training module shall clean and prepare the dataset.
3. The training module shall create a machine learning pipeline for funding prediction.
4. The training module shall save the trained model and metadata for later use.

### 2.2 External Interface Requirements

#### 2.2.1 User Interface
The system provides two user interfaces:

1. A static web interface built with HTML, CSS, and JavaScript.
2. A Streamlit-based prototype interface.

The interface shall allow users to:

1. Enter startup details.
2. Submit the input for analysis.
3. View startup profile results.
4. View AI similarity results.
5. View similar startups.
6. View funding analysis and predicted funding.

#### 2.2.2 Software Interface
1. The frontend shall communicate with the backend through HTTP requests.
2. The backend shall use the startup CSV dataset.
3. The backend shall interact with Ollama through its local HTTP API.
4. The backend shall load trained model files from local storage.

#### 2.2.3 Hardware Interface
No special hardware is required beyond a normal computer system. Better performance may be observed on systems with higher RAM and processing power.

#### 2.2.4 Communication Interface
1. The frontend and backend shall communicate through REST API calls.
2. The backend and Ollama service shall communicate through local HTTP requests.

### 2.3 Non-Functional Requirements

#### 2.3.1 Performance Requirements
1. The system should return results within a reasonable amount of time for a single user request.
2. The similarity search should be fast after the model and index are loaded.
3. The initial startup time may be higher because embeddings and models are loaded into memory.

#### 2.3.2 Reliability Requirements
1. The system should handle invalid input gracefully.
2. The system should show proper error messages if Ollama is not running.
3. The system should show proper error messages if the funding model is missing.
4. The backend should not crash for normal input errors.

#### 2.3.3 Usability Requirements
1. The interface should be simple and easy to understand.
2. The user should be able to complete the evaluation process without technical knowledge.
3. The displayed outputs should be readable and clearly separated into sections.

#### 2.3.4 Maintainability Requirements
1. The codebase should remain modular with separate files for frontend, backend, AI integration, and ML training.
2. The system should allow future addition of more features and better models.
3. The dataset and model files should be replaceable without major code changes.

#### 2.3.5 Portability Requirements
1. The project should be runnable on different operating systems with Python support.
2. The system should work in any modern browser for the web interface.

#### 2.3.6 Security Requirements
1. The system should validate user input before processing.
2. The current prototype is intended for local use and does not implement full production security.
3. Future versions should include authentication, stricter CORS configuration, and deployment-level security.
