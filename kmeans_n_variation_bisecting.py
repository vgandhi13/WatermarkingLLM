import os
import sys
import time

import joblib
import nltk
import numpy as np
from gensim.models import FastText
import datasets
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans, BisectingKMeans



def process_data(window_size=3):
    # Define the file path


    # Prepare corpus (tokenized sentences)
    corpus = []

    dataset = datasets.load_dataset("BAAI/Infinity-Instruct", '3M', split='train')
    data = list(dataset['conversations'])[:1000]
    corpus = []
    for conversation in data:
        conversation_text = conversation[1]['value'] #get the gpt response part
        corpus.append(conversation_text)
        
    # Train FastText model on the corpus
    sys.stdout.flush()
    start = time.time()
    fasttext_model = FastText.load_fasttext_format("cc.en.300.bin")
    end = time.time()
    print("FastText model loaded in {:.2f} seconds".format(end - start))
    sys.stdout.flush()
    # Collect all 3-token windows
    three_token_windows = []

    for text in corpus:
        for i in range(len(text) - window_size - 1):
            window = text[i : i + window_size]
            three_token_windows.append(window)

    # Generate embeddings for each window
    window_embeddings = []

    for window in three_token_windows:
        # start = time.time()
        embeddings = [fasttext_model.wv[token] for token in window]
        # end = time.time()
        # print("Time taken to get embeddings for window {}: {:.2f} seconds".format(window, end - start))
        # sys.stdout.flush()
        # HERE WE ADD NOISE TO THE EMBEDDINGS!!!
        noise = np.random.normal(0, 0.01, len(embeddings[0])) #this draws from a normal distribution
        embeddings = [embedding + noise for embedding in embeddings]
        avg_embedding = sum(embeddings) / len(embeddings)
        window_embeddings.append(avg_embedding)

    # Example: Print the first window and its embedding
    print("First window:", three_token_windows[0])
    print("First embedding (first 10 dims):", window_embeddings[0][:10])  # Print first 10 dims
    print("Number of windows:", len(three_token_windows))
    return window_embeddings


def train_kmeans(embeddings, n_clusters=[2040]):#[160, 250, 540, 1020, 2040]): # take 2040 for clusters and then mod the index during sampling
    ############################################################################
    """
    Train KMeans clustering on the given embeddings.
    
    Args:
        embeddings (list): List of embeddings to cluster.
        n_clusters (int): Number of clusters for KMeans.
        
    Returns:
        kmeans (KMeans): Trained KMeans model.
    """
    X = np.array(embeddings)

    kmeans = []



    for n_cluster in n_clusters:
        start = time.time()
        print("Starting KMeans initialization...")
        sys.stdout.flush()
        print("Training KMeans with n_clusters = ", n_cluster)
        sys.stdout.flush()
        # bisection strategy should be the largest cluster 
        kmeans.append(BisectingKMeans(n_clusters=n_cluster, random_state=42, verbose=3, bisecting_strategy='largest_cluster'))
        kmeans[-1].fit(X) 
        end = time.time()
        print("KMeans training completed in {:.2f} seconds".format(end - start))
        sys.stdout.flush()
    
    return kmeans


if __name__ == "__main__":
    embeddings = process_data()
    kmeans_models = train_kmeans(embeddings)
    

    for i, kmeans_model in enumerate(kmeans_models):
        print(f"KMeans model trained with {kmeans_model.n_clusters} clusters.")
        joblib.dump(kmeans_model, f"kmeans_model_{kmeans_model.n_clusters}_n3_bisecting.pkl")
        print(f"KMeans model saved as kmeans_model_{kmeans_model.n_clusters}_n3_bisecting.pkl")
    
        kmeans_model = joblib.load(f"kmeans_model_{kmeans_model.n_clusters}_n3_bisecting.pkl")
        print(f"KMeans model loaded from kmeans_model_{kmeans_model.n_clusters}_n3_bisecting.pkl")    
        print("Cluster centers (first 10 dims):", kmeans_model.cluster_centers_[:10])
        print("Number of clusters:", kmeans_model.n_clusters)
