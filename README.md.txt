# Simple-Personal-Assistant
 A simple personal assistant chatbot built with Python and Scikit-Learn.
# Project Repository Link: https://github.com/NasimCZD/Simple-Personal-Assistant.git

# How to Run the Chatbot (REquired Steps)
 To ensure the project runs without errors, follow these steps exactly:
 # Step1: Clone and Set Up the environment
 1. Install Git: If you don't have it, install Git from git-scm.com
 2. Clone the REpository: Open your terminal/CMD, navigate to where you want to save the project and run:
      git clone [https://github.com/NasimCZD/Simple-Personal-Assistant.git]
      cd Simple-Personal-Assistant
3. Create a Virtual Machine: python -m venv chatbot_env
4. Activate the environment: chatbot_env\Scripts\activate

# Step2. Install Libraries
You must install all the project dependencies (scikit-learn, nlkt, and numpy)
Run: pip install scikit-learn numpy nlkt

# Step 3: Download NLTK Data (Crucial Fix)
1. Create Data Folder: In the project root (Simple-Personal-Assistant), create a folder named nltk_data
2. Download Data: Run the following command to active the virtual environment
    python
   import nltk
   download_dir - "nltk_data" #relative path to the folder you just created
   nltk.download('punkt'), download_dir=download_dir)
   nltk.download('wordnet'), download_dir=download_dir)
   nltk.download('punkt_tab'), download_dir=download_dir)
   exit()

# Step4: Run the Chatbot
Now you can run the main script. The script will train the model, save the necessary files, and start the chat loop. Run the following
  python assistant.py

When prompted, try phrases like "Hi", "What time it is".

