import numpy as np
import joblib
import datasets
import pickle
from gensim.models import FastText
from ecc.ciphertext import McEliece
from huggingface_hub import login
import os
from dotenv import load_dotenv

fasttext_model = FastText.load_fasttext_format("cc.en.300.bin")
load_dotenv()
login(token = os.getenv('HF_TOKEN'))
dataset = datasets.load_dataset("BAAI/Infinity-Instruct", '3M', split='train')
class SimpleKMeansWrapper:
    """
    Simple wrapper that replaces cluster IDs with count-based labels.
    """
    
    def __init__(self, kmeans_model_path, window_size=3, batch_messages=None, crypto_scheme='Ciphertext'):
        # Load the original KMeans model
        self.kmeans_model = joblib.load(kmeans_model_path)
        
        # Load FastText model
        self.fasttext_model = fasttext_model
        
        # Generate codewords like in batch_warper.py
        codewords = []
        if crypto_scheme == 'McEliece':
            mceliece = McEliece()
            for message in batch_messages:
                # Convert message to binary string (8 bits per character)
                binary_message = ''.join(format(ord(c), '08b') for c in message)
                # Pad or truncate to k bits (8 bits for RM(1,7))
                if len(binary_message) > mceliece.k:
                    binary_message = binary_message[:mceliece.k]
                else:
                    binary_message = binary_message.ljust(mceliece.k, '0')
                
                # Encrypt using McEliece
                codeword = mceliece.encrypt(binary_message)
                codewords.append("codeword: " + codeword)
        elif crypto_scheme == 'Ciphertext':
            for message in batch_messages:
                mceliece = McEliece()
                # Convert message to binary string
                binary_message = ''.join(format(ord(c), '08b') for c in message)
                if len(binary_message) > mceliece.k:
                    binary_message = binary_message[:mceliece.k]
                else:
                    binary_message = binary_message.ljust(mceliece.k, '0')
                codeword = mceliece.encrypt(binary_message)
                codewords.append(codeword)
        
        # Get codeword length from the first codeword
        self.codeword_length = len(codewords[0])

    
        # Get both instructions and outputs
        data = list(dataset['conversations'])[:1000]
        corpus = []
        for conversation in data:
            conversation_text = conversation[1]['value'] #get the gpt response part
            corpus.append(conversation_text)
        
        cluster_counts = {i: 0 for i in range(2040)}

        n_grams = set()
        for text in corpus:
            for i in range(len(text) - window_size - 1):
                n_gram = text[i:i+window_size]
                embeddings = [self.fasttext_model.wv[token] for token in n_gram]
                avg_embedding = sum(embeddings) / len(embeddings)
                avg_embedding = np.array(avg_embedding, dtype=float).reshape(1, -1)
                cluster = self.kmeans_model.predict(avg_embedding)[0]
                cluster_counts[cluster] += 1

        # print(cluster_counts)
        # need to get the cluster counts mod 128
        
        
        # Sort by count, greatest to least 
        sorted_clusters = dict(sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True))
        print('SORTED CLUSTERS: ', sorted_clusters)
        self.sorted_clusters = sorted_clusters
        self.mapping = {}
        for i, cluster in enumerate(sorted_clusters.keys()):
            self.mapping[i] = cluster
        

        cluster_counts_mod = {i: 0 for i in range(self.codeword_length)}
        for cluster, index in self.mapping.items():
            cluster_counts_mod[ cluster % self.codeword_length] += sorted_clusters[index]
        print("Cluster counts mod 128 method 1: ", cluster_counts_mod)
        cluster_counts_zigzag = {i: 0 for i in range(self.codeword_length)}
        # zigzag metho
        direction = 1
        new_index = 0
        for cluster, index in self.mapping.items():
            cluster_counts_zigzag[new_index] += sorted_clusters[index]
            if direction == 1:
                new_index += 1
            else:
                new_index -= 1
            if new_index == 128:
                new_index = 127
                direction = -1
            if new_index == -1:
                new_index = 0
                direction = 1

        print("Cluster counts mod 128 method 2: ", cluster_counts_zigzag)
        # print(self.mapping)
        
        # for cluster in sorted_clusters:
        #     zigzag_mapping[cluster % self.codeword_length].append(cluster)
        # print(zigzag_mapping)
        

        # print(self.mapping)



    
    def predict(self, X):
        """
        Predict clusters using count-based labels instead of original cluster IDs.
        """
        
        
        if not hasattr(self, 'kmeans_model') or self.kmeans_model is None:
            self.kmeans_model = joblib.load("kmeans_model3.pkl")
        cluster = self.kmeans_model.predict(X)[0]
        
        # Look up each cluster in the mapping and return the codeword index
        result = self.mapping[cluster] % self.codeword_length
        # print(result)
        
        return result 
    
    def save_to_pickle(self, filepath):
        """
        Save the wrapper instance to a pickle file.
        
        Args:
            filepath (str): Path where to save the pickle file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.mapping, f)
        print(f"Wrapper saved to {filepath}")
    


def load_wrapper_from_pickle(filepath):
    """
    Load a saved wrapper instance from a pickle file.
    
    Args:
        filepath (str): Path to the pickle file
        
    Returns:
        SimpleKMeansWrapper: The loaded wrapper instance
    """
    with open(filepath, 'rb') as f:
        mapping = pickle.load(f)
    wrapper = SimpleKMeansWrapper.__new__(SimpleKMeansWrapper)
    wrapper.mapping = mapping
    wrapper.codeword_length = 128
    return wrapper


if __name__ == "__main__":
    batch_messages = ["test message 1", "test message 2"]
    # wrapper = SimpleKMeansWrapper("kmeans_model_2040_n3.pkl", window_size=3, batch_messages=batch_messages)

        
    # # Save the wrapper to pickle
    # wrapper.save_to_pickle("saved_wrapper_kmeans_2040_n3.pkl")
    

    embeddings = [fasttext_model.wv[token] for token in ["that", "is", "crazy"]]
    avg_embedding = sum(embeddings) / len(embeddings)
    # Reshape to (1, -1) since predict expects a 2D array
    avg_embedding = np.array(avg_embedding).reshape(1, -1)
    # idx = wrapper.predict(avg_embedding)
    # print(f"Test prediction result: {idx}")
    
    # # Demonstrate loading from pickle
    print("\n--- Loading from pickle ---")
    loaded_wrapper = load_wrapper_from_pickle("saved_wrapper_kmeans_2040_n3.pkl")
    idx_loaded = loaded_wrapper.predict(avg_embedding)
    print(loaded_wrapper.mapping)
    print(f"Loaded wrapper prediction result: {idx_loaded}") 
        

