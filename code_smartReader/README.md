# SmartReader - BGSW FIT.Fest GenAI Hackathon

## About

This is the submission of team **Pheonix** for PHASE 2 of the BGSW FIT.Fest GenAI Hackathon.
A RAG Q&A application to chat with PDF files, also containing images and tables.

## IMPORTANT NOTE: In the video, at timestamp 12:48, we have misspoke Phase 3 as Approach 3. It should have been Phase 3.

## KPI's acheived from all phases

## 🕹️ - Installation Instructions

> Prerequisites: Python, pip

1. Navigate to the folder "code_smartReader"

2. Create a virtual environment with either _conda_ or _python_
  
  - With python:
    
    - pip install virtualenv
    - virtualenv -p python3.11 <venv name>
    
    -> for linux:    
      - source ./<venv name>/bin/activate
    -> for windows:
      - source ./<venv name>/Scripts/activate.bat

   - With conda:
    - conda init
    - conda create -n <venv name> python=3.11.9
    - conda activate <venv name>

3. Install the required libraries

  - pip install --no-deps -r requirements.txt

4. Create a .env file in the root directory. Obtain the following API keys and add them in the .env file. Follow syntax of .env.example for reference.

- GOOGLE_API_KEY:
  - https://aistudio.google.com/app/u/1/apikey.

  - NOTE: An API key is provided in the .env file for quick use.

5. Start the streamlit file

  - streamlit run main.py

## PLEASE GO THROUGH PPT AND VIDEO FOR DETAILED EXPLAINATION.

## STOP THE CODE AND RERUN TO TRY OUT APPROACH 2.

## 🕹️ - For code analysis

> Go through files for understanding of code. Comments added at places to aid in this process.

## 👥 - TEAM DETAILS

Institution - KLE Technological University

- Pranav S Bhat - 8861668850 - 01fe21bcs230@kletech.ac.in
- Ravishankar B - 6363663113 - 01fe21bcs177@kletech.ac.in
- Kushalgouda S Patil - 8105784976 - kspatil.ksp@gmail.com
