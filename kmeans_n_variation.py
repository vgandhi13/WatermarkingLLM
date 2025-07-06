import os
import sys
import time

import joblib
import nltk
import numpy as np
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans, MiniBatchKMeans

nltk.download('punkt_tab')

def process_data(window_size=5):
    # Define the file path
    file_path = "openwebtext_2017_18_1e5"

    # Prepare corpus (tokenized sentences)
    corpus = []

    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            # lines = [f.readline() for _ in range(1000)] # replace with f.readlines()
            lines = f.readlines()
            print('Number of lines: ', len(lines))

            for i, line in enumerate(lines):
                # Add '...' to the beginning
                
                if line == '\n':
                    continue
                line = '. . . ' + line.strip()

                # Tokenize and add to corpus
                tokens = word_tokenize(line)
                corpus.append(tokens)  # Each line as a tokenized sentence
                    
    # Train FastText model on the corpus
    sys.stdout.flush()
    start = time.time()
    fasttext_model = FastText.load_fasttext_format("cc.en.300.bin")
    end = time.time()
    print("FastText model loaded in {:.2f} seconds".format(end - start))
    sys.stdout.flush()
    # Collect all 3-token windows
    three_token_windows = []

    for tokens in corpus:
        for i in range(len(tokens) - window_size - 1):
            window = tokens[i : i + window_size]
            three_token_windows.append(window)

    # Generate embeddings for each window
    window_embeddings = []

    for window in three_token_windows:
        # start = time.time()
        embeddings = [fasttext_model.wv[token] for token in window]
        # end = time.time()
        # print("Time taken to get embeddings for window {}: {:.2f} seconds".format(window, end - start))
        # sys.stdout.flush()
        avg_embedding = sum(embeddings) / len(embeddings)
        window_embeddings.append(avg_embedding)

    # Example: Print the first window and its embedding
    print("First window:", three_token_windows[0])
    print("First embedding (first 10 dims):", window_embeddings[0][:10])  # Print first 10 dims
    print("Number of windows:", len(three_token_windows))
    return window_embeddings


def train_kmeans(embeddings, n_clusters=[160, 250, 540, 1020, 2040]): # take 2040 for clusters and then mod the index during sampling
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

    # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=10)
    kmeans = []

    for n_cluster in n_clusters:
        start = time.time()
        print("Starting KMeans initialization...")
        sys.stdout.flush()
        print("Training KMeans with n_clusters = ", n_cluster)
        sys.stdout.flush()
        kmeans.append(MiniBatchKMeans(n_clusters=n_cluster, random_state=42, batch_size=1000, verbose=3, max_iter = 1))
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
        joblib.dump(kmeans_model, f"kmeans_model_{kmeans_model.n_clusters}_n5.pkl")
        print(f"KMeans model saved as kmeans_model_{kmeans_model.n_clusters}_n5.pkl")
    
        kmeans_model = joblib.load(f"kmeans_model_{kmeans_model.n_clusters}_n5.pkl")
        print(f"KMeans model loaded from kmeans_model_{kmeans_model.n_clusters}_n5.pkl")    
        print("Cluster centers (first 10 dims):", kmeans_model.cluster_centers_[:10])
        print("Number of clusters:", kmeans_model.n_clusters)
