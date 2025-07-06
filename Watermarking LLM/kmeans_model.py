import os
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import joblib
from gensim.models import FastText
import time
import sys


def process_data():
    # Define the file path
    file_path = "openwebtext_2017_18_1e5"

    # Prepare corpus (tokenized sentences)
    corpus = []

    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            # Read first 10 lines
            #lines = [f.readline() for _ in range(20)]  # replace with f.readlines()
            lines = f.readlines()
            print('Number of lines: ',len(lines))
            for i, line in enumerate(lines):
                # Add '...' to the beginning
                
                if line == '\n':
                    print('entered')
                    continue
                line = '. . . ' + line.strip()


                # Tokenize and add to corpus
                tokens = word_tokenize(line)
                corpus.append(tokens)  # Each line as a tokenized sentence
            

    # Train FastText model on the corpus
    print('reached here')
    sys.stdout.flush()
    start = time.time()
    fasttext_model = FastText.load_fasttext_format("cc.en.300.bin")
    end = time.time()
    print("FastText model loaded in {:.2f} seconds".format(end - start))
    print('reached here2')
    sys.stdout.flush()
    # Collect all 3-token windows
    three_token_windows = []

    for tokens in corpus:
        for i in range(len(tokens) - 2):
            window = tokens[i:i+3]
            three_token_windows.append(window)

    # Generate embeddings for each window
    window_embeddings = []

    for window in three_token_windows:
        start = time.time()
        embeddings = [fasttext_model.wv[token] for token in window]
        end = time.time()
        print("Time taken to get embeddings for window {}: {:.2f} seconds".format(window, end - start))
        sys.stdout.flush()
        avg_embedding = sum(embeddings) / len(embeddings)
        window_embeddings.append(avg_embedding)

    # Example: Print the first window and its embedding
    print("First window:", three_token_windows[0])
    print("First embedding (first 10 dims):", window_embeddings[0][:10])  # Print first 10 dims
    print("Number of windows:", len(three_token_windows))
    return window_embeddings

from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import joblib

def train_kmeans(embeddings, n_clusters=2040): # take 2040 for clusters and then mod the index during sampling
    ############################################################################
    """
    Train KMeans clustering on the given embeddings.
    
    Args:
        embeddings (list): List of embeddings to cluster.
        n_clusters (int): Number of clusters for KMeans.
        
    Returns:
        kmeans (KMeans): Trained KMeans model.
    """
    # Convert embeddings list to a NumPy array
    X = np.array(embeddings)
    
    # Train KMeans
    start = time.time()
    print("Starting KMeans initialization...")
    sys.stdout.flush()
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=10)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, verbose=3,max_iter = 1)
    print("Starting KMeans Training...")
    sys.stdout.flush()
    kmeans.fit(X)
    end = time.time()
    print("KMeans training completed in {:.2f} seconds".format(end - start))
    sys.stdout.flush()
    # # Example: Print first 10 cluster labels
    # labels = kmeans.labels_
    # print("First 10 cluster labels:", labels[:10])
    
    return kmeans


if __name__ == "__main__":
    # # Process data and train KMeans
    embeddings = process_data()
    kmeans_model = train_kmeans(embeddings)
    
    # # Save the model
    joblib.dump(kmeans_model, "kmeans_model4.pkl")
    print("KMeans model saved as kmeans_model4.pkl")
    # Load the model
    kmeans_model = joblib.load("kmeans_model4.pkl")
    print("KMeans model loaded from kmeans_model4.pkl")
    # Example: Print the cluster centers
    print("Cluster centers (first 10 dims):", kmeans_model.cluster_centers_[:10])  # Print first 10 dims
    #Example: Print the number of clusters
    print("Number of clusters:", kmeans_model.n_clusters)
