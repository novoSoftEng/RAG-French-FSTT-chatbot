# Fine-Turn and RAG French FSTT chatbot

This project is about create a chatbot about FSTT (Faculty of Science and Thechnology in Tanger) in french language using Fine-turn and Retrieval Augmented Generation (RAG) techniques with User Interface :

- Frontend and Backend Development: Seamless integration ensuring a smooth user experience.
- Advanced NLP Features: Utilizing powerful tools such as Python, BeautifulSoup, Chroma Database, Gemma 2b LLM, and LangChain Framework.

This README provides instructions on how to set up and run the Chroma Vector database using Docker, how to Create *RAG* System for **Gemma 2b LLM**, how to *Fine-Turn* Gemma 2b LLM and how to run the chatbot interface.

## Prerequisites :

- **Docker** and **Python** must be installed on your system.
- Ensure you have internet access for downloading the necessary Docker image and model.

## Running the Project :

### Install the Project :

   ```sh
   git clone https://github.com/novoSoftEng/RAG-French-FSTT-chatbot.git
   cd RAG-French-FSTT-chatbot
   ```

### Set up the requisites :

- run cmd (need good Internet):
  ```sh
   docker compose up
   ```

### Create Chroma Vector Database :

To save data to the Chroma Vector database, use the provided Jupyter Notebook.

1. Open the Jupyter Notebook located at `Data/save_to_chroma.ipynb`.

2. Follow the instructions within the notebook to save your data to the Chroma Vector database.

### Create RAG System :


### Fine-Turn the LLM :


---

This README file should help you get started with running and using the Chroma Vector database and chatbot app.
