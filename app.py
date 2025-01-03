import pandas as pd
import chardet
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from textblob import TextBlob
import streamlit as st
from prettytable import PrettyTable

def detect_encoding(file_path):
    rawdata = open(file_path, "rb").read()
    result = chardet.detect(rawdata)
    return result['encoding']

def load_data(file_path):
    st.write("Loading the dataset...")
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        st.write("Dataset loaded successfully with UTF-8 encoding.")
        return data
    except UnicodeDecodeError:
        st.write("UnicodeDecodeError: Trying with ISO-8859-1 encoding...")
    
    try:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
        st.write("Dataset loaded successfully with ISO-8859-1 encoding.")
        return data
    except UnicodeDecodeError:
        st.write("ISO-8859-1 failed. Detecting encoding...")
    
    encoding = detect_encoding(file_path)
    st.write(f"Detected encoding: {encoding}")
    data = pd.read_csv(file_path, encoding=encoding)
    st.write(f"Dataset loaded successfully with detected encoding {encoding}.")
    return data

#Preprocess the data
def preprocess_data(data):
    st.write("Preprocessing data...")
    data['description'] = data['description'].fillna('')
    data['title'] = data['title'].fillna('')
    data['curriculum'] = data['curriculum'].fillna('Curriculum details not available')  
    data['instructor'] = data['instructor'].fillna('Instructor details not available')
    data['combined_text'] = data['title'] + " " + data['description']
    st.write("Data preprocessing completed.")
    return data

def generate_embeddings(data):
    st.write("Generating embeddings...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    data['embedding'] = data['combined_text'].apply(lambda x: embedding_model.encode(x))
    st.write("Embeddings generated successfully.")
    return data, embedding_model

def create_faiss_index(data):
    dimension = len(data['embedding'][0]) 
    index = faiss.IndexFlatL2(dimension)  
    embeddings = np.vstack(data['embedding'].values) 
    index.add(embeddings) 
    st.write("Index created successfully.")
    return index

def correct_spelling(query):
    corrected_query = str(TextBlob(query).correct())
    if corrected_query != query:
        st.write(f"Did you mean: '{corrected_query}'?")
    return corrected_query

def search_courses(query, top_k=5, index=None, data=None, embedding_model=None):
    query_embedding = embedding_model.encode(query) 
    distances, indices = index.search(np.array([query_embedding]), top_k)  
    results = []
    for idx in indices[0]:
        if idx < len(data):
            course = data.iloc[idx]
            results.append({
                'title': course['title'],
                'description': course['description'],
                'curriculum': course['curriculum'],  
                'instructor': course['instructor'],
                'distance': distances[0][list(indices[0]).index(idx)]
            })
    return results

def display_results(results):
    if results:
        table = PrettyTable()
        table.field_names = ["Rank", "Title", "Description", "Curriculum", "Instructor", "Distance"]
        for i, result in enumerate(results):
            table.add_row([i + 1, result['title'], result['description'], result['curriculum'], result['instructor'], f"{result['distance']:.4f}"])
        st.write(table)
    else:
        st.write("No results found for your query. Try another search.")
def main():
    file_path = "analyticsvidhya_courses_full_simulated.csv"
    data = load_data(file_path)
    data = preprocess_data(data)
    data, embedding_model = generate_embeddings(data)
    index = create_faiss_index(data)
    st.write("Welcome to the Course Search System!")
    st.write("You can search for courses based on course titles and descriptions.")
    query = st.text_input("Enter your search query:", "")
    
    if query:
        corrected_query = correct_spelling(query)
        results = search_courses(corrected_query, top_k=5, index=index, data=data, embedding_model=embedding_model)
        st.write("\nSearch Results:")
        display_results(results)

if __name__ == "__main__":
    main()
