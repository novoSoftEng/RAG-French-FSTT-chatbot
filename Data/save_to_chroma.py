#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
# pd.set_option('display.width', 1000) 
# pd.set_option('display.max_colwidth', 1000) 
import numpy as np
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from chromadb.config import Settings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from langdetect import detect
import nltk
import re
import chromadb
import string
import uuid


# In[3]:

nltk.download('stopwords')
french_stopwords = set(stopwords.words('french'))


# In[4]:


article = pd.read_csv("fstt-articles.csv")
clubs = pd.read_csv("fstt-clubs-info.csv")
dep = pd.read_csv("fstt-departements-info.csv")
formation = pd.read_csv("fstt-formation-initial.csv")
general_info = pd.read_csv("general_info.csv")
general_info.columns=["info", "fstt"]


# In[5]:


clubs.rename(columns={'departement_info': 'club_description'}, inplace=True)


# In[6]:


article.head(3)


# In[7]:


# Create a new column by merging all columns with a formatted string
article['merged_content'] = article.apply(
    lambda row: f"titre de post ou article : {row['post_title']} | contenu post ou article : {row['post_content']} | lien de post ou article : {row['post_link']}",
    axis=1
)


# In[8]:


# Display the DataFrame
print(article['merged_content'][0])


# In[9]:


# Create a new column by merging all columns with a formatted string
clubs['merged_content'] = clubs.apply(
    lambda row: f"nom du club : {row['club_name']} | description du club  : {row['club_description']} | lien du club : {row['club_link']}",
    axis=1
)


# In[10]:


clubs['merged_content'][0]


# In[11]:


# Remove newline characters from 'departement_info' column
dep['departement_info'] = dep['departement_info'].str.replace('\n', ' - ')


# In[12]:


dep.head()


# In[13]:


# Create a new column by merging all columns with a formatted string
dep['merged_content'] = dep.apply(
    lambda row: f"nom de departement : {row['departement_name']} | le nom de Coordinnateur avec son email : {row['departement_info']}",
    axis=1
)

# Create a new column by merging all columns with a formatted string
formation['merged_content'] = formation.apply(
    lambda row: f"nom de formation : {row['mst_name']} | type de formation  : {row['formation_type']} | objective de formation  : {row['mst_objectif']} | programme de formation : {row['mst_program']} | Coordinnateur de formation : {row['mst_Coord']} | link de formation : {row['mst_link']} |",
    axis=1
)



# Remove newline characters from 'departement_info' column
general_info['info'] = general_info['info'].str.replace('\n', '  ')


# In[20]:


def is_french(text):
    try:
        return detect(text) == 'fr'
    except:
        return False
def preprocessing(text):
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    # text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions
    text = re.sub(r'@\w+', ' ', text)

    text = re.sub(r'\xa0', ' ', text)
    
    # Remove \n
    text = re.sub(r'\n', ' ', text)
    
    # Remove specific unwanted characters
    text = re.sub(r'«|»|“|”|’|‘', ' ', text)
    
        # Tokenization
    """tokens = sent_tokenize(text, language="french")
    print("Tokenization :" ,tokens)"""
    
    # Check if text is in French
    if not is_french(text):
        return ''
        
    # Remove punctuation
    # text = text.translate(str.maketrans("", "", string.punctuation))
    
    # # Tokenization using RegexpTokenizer
    # tokenizer = RegexpTokenizer(r'\w+')
    # tokens = tokenizer.tokenize(text)
    
    # tokens = word_tokenize(text, language="french")
    # tokens = [word for word in tokens if word not in french_stopwords]
    
    
    # Stemming
    # stemmer =nltk.stem.snowball.FrenchStemmer()
    # tokens_stemmed = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back into a single string
    return text


# In[21]:


article["merged_content"] = article["merged_content"].apply(preprocessing)
clubs["merged_content"] = clubs["merged_content"].apply(preprocessing)
formation["merged_content"] = formation["merged_content"].apply(preprocessing)
dep["merged_content"] = dep["merged_content"].apply(preprocessing)
general_info['info'] = general_info['info'].apply(preprocessing)

article.drop(columns="No", inplace=True)
clubs.drop(columns="No", inplace=True)
formation.drop(columns="No", inplace=True)
dep.drop(columns="No", inplace=True)


# In[25]:



client = chromadb.HttpClient(host='localhost', port=8000)


# In[26]:




# In[27]:


# Create a collection in ChromaDB with the OllamaEmbeddingFunction
collection = client.create_collection(name="text_embeddings")


# In[28]:


import uuid

def process_and_store_embeddings(dataframe, column_names, collection):
    ids = []
    metadatas = []
    documents = []

    for idx, row in dataframe.iterrows():
        doc_metadata = {}
        for key, value in row.items():
            if key not in column_names:
                doc_metadata[key] = value  # Leave non-string fields as is
        
        for column in column_names:
            sentence = row[column]
            if sentence is not None and sentence != '':
                ids.append(str(uuid.uuid1()))
                metadatas.append(doc_metadata)
                documents.append(sentence)
    collection.add(
        ids=ids,
        #embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

# Assuming article, clubs, formation DataFrames are already preprocessed
# Store embeddings, metadata, and documents from each DataFrame
process_and_store_embeddings(article, ["merged_content"], collection)
process_and_store_embeddings(clubs, ["merged_content"], collection)
process_and_store_embeddings(formation, ["merged_content"], collection)
process_and_store_embeddings(dep, ["merged_content"], collection)
process_and_store_embeddings(general_info, ["info"], collection)

print("Embeddings, metadata, and documents stored in ChromaDB successfully.")


