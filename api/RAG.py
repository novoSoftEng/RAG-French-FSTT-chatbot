from transformers import AutoTokenizer, AutoModelForCausalLM
import chromadb
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re
from IPython.display import display, Markdown
import pandas as pd


# ## AI Agent class :

# In[3]:


import os
def extract_answer_from_text(text):
    answer_match = re.search(r'Answer:\s*(.*)', text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""

    return answer
# Set your Hugging Face API token
HUGGINGFACE_HUB_TOKEN = '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'
os.environ['HUGGINGFACE_HUB_TOKEN'] = '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'


# In[13]:


# model_name="google/gemma-2b-it"
# model_name="aymanboufarhi/gemma-fstt"

class AIAgent:
    
    def __init__(self, model_name="aymanboufarhi/gemma2B-chat-bot-fstt", max_length=1000):
        self.max_length = max_length
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_HUB_TOKEN)
            self.gemma_llm = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGINGFACE_HUB_TOKEN)
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def create_prompt(self, query, context):
        # Prompt template
        prompt = f"""
        You are an AI Agent specialized to answer questions about FSTT (faculty of science and technology in Tanger).
        Explain the concept or answer the question about FSTT.
        In order to create the answer,use the information from the context if it seems to be relevant to the question provided (Context). 
        and the context will be as a list, so you must use just the most relevent informations from the list.
        Answer with simple words.
        If needed, include also explanations.
        it's important to answer with french languge.
        return only the answer
        Question: {query}
        Context: {context}
        Answer:
        """
        return prompt
    
    def generate(self, query, retrieved_info=None):
        prompt = self.create_prompt(query, retrieved_info)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        # Answer generation
        answer_ids = self.gemma_llm.generate(input_ids, max_new_tokens=self.max_length)
        
        # Decode and return the answer
        answer = self.tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        
        return answer

# ### Test the AIAgent :

# In[14]:






# In[30]:
def connect(): 
    # Configure the ChromaDB client with persistence
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_collection(name="text_embeddings")
    return collection

class RAGSystem:
    """Sentence embedding based Retrieval Based Augmented generation.
       Given a ChromaDB collection, retriever finds num_retrieved_docs relevant documents."""
    
    def __init__(self, ai_agent, collection=connect(), num_retrieved_docs=2):
        self.num_docs = num_retrieved_docs
        self.collection = collection
        self.ai_agent = ai_agent
    
    def retrieve(self, query):
        # Retrieve top k similar documents to query
        results = self.collection.query(query_texts=[query], n_results=self.num_docs)
        docs = [result for result in results['documents']]
        return docs
    
    def query(self, query):
        # Generate the answer
        context_docs = self.retrieve(query)
        context_docs = context_docs[0]
        print(context_docs)
        
        answer = self.ai_agent.generate(query, context_docs)

        
        return answer

# In[8]:


def colorize_text(text):
    for word, color in zip(["Question", "Prompt", "Answer", "Context"], ["blue", "magenta", "red", "green"]):
        text = text.replace(f"\n\n{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text


# In[31]:









# In[34]:


# query = '''<|system|>FSTT c'est la Faculté des Sciences et Techniques de Tanger 
# <|user|> Donne le nombre de départements avec les noms et informations de chaque departement
# <|assistant|>'''




