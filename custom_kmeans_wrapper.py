import numpy as np
import joblib
import datasets
from gensim.models import FastText
from ecc.mceliece import McEliece
from ecc.ciphertext import Ciphertext

fasttext_model = FastText.load_fasttext_format("cc.en.300.bin")
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
            for message in batch_messages:
                codeword = McEliece().encrypt(message.encode('utf-8'))[0]
                E = ''.join(format(byte, '08b') for byte in codeword)
                codewords.append("codeword: " + E)
        elif crypto_scheme == 'Ciphertext':
            for message in batch_messages:
                ciphertext = Ciphertext()
                codeword = ciphertext.encrypt('Asteroid')
                codewords.append(codeword)
        
        # Get codeword length from the first codeword
        self.codeword_length = len(codewords[0])

        dataset = datasets.load_dataset("BAAI/Infinity-Instruct", '3M')
    
        # Get both instructions and outputs
        corpus = []
        for conversation in dataset['train']['conversations'][:100]:
            conversation_text = conversation[1]['value'] #get the gpt response part
            corpus.append(conversation_text)
        
        cluster_counts = {}

        n_grams = set()
        for text in corpus:
            for i in range(len(text) - window_size - 1):
                n_gram = text[i:i+window_size]
                if n_gram not in n_grams:
                    n_grams.add(n_gram)
                embeddings = [self.fasttext_model.wv[token] for token in n_gram]
                avg_embedding = sum(embeddings) / len(embeddings)
                # Convert to numpy array like in original training (this converts float32 to float64)
                avg_embedding = np.array(avg_embedding).reshape(1, -1)
                cluster = self.kmeans_model.predict(avg_embedding)[0]
                if cluster not in cluster_counts:
                    cluster_counts[cluster] = 1
                else:
                    cluster_counts[cluster] += 1



        
        # Sort by count (most n-grams first) and assign labels 0, 1, 2, ...
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create mapping using back-and-forth method
        self.mapping = {i: set() for i in range(self.codeword_length)} 
        direction = 1
        i = 0
        
        for cluster, count in sorted_clusters:
            if direction == 1:
                self.mapping[i].add(cluster)
                i += 1
                if i >= self.codeword_length:
                    i = self.codeword_length - 1
                    direction = -1
            else:
                self.mapping[i].add(cluster)
                i -= 1
                if i < 0:
                    i = 0
                    direction = 1
        print(self.mapping)



    
    def predict(self, X):
        """
        Predict clusters using count-based labels instead of original cluster IDs.
        """
        
        
        # Get original predictions
        cluster = self.kmeans_model.predict(X)[0]
        print(cluster)
        
        # Look up each cluster in the mapping and return the codeword index
        result = -1
        for ind in self.mapping:
            if cluster in self.mapping[ind]:
                result = ind
            print(ind, len(self.mapping[ind]))
        
        return result 
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original KMeans model."""
        return getattr(self.kmeans_model, name)



if __name__ == "__main__":
    # Test the wrapper
        batch_messages = ["test message 1", "test message 2"]
        wrapper = SimpleKMeansWrapper("kmeans_model3.pkl", window_size=3, batch_messages=batch_messages)
        
        # Test with dummy embedding 
        fasttext_model = FastText.load_fasttext_format("cc.en.300.bin")
        embeddings = [fasttext_model.wv[token] for token in ["test", "message", "1"]]
        avg_embedding = sum(embeddings) / len(embeddings)
        # Reshape to (1, -1) since predict expects a 2D array
        avg_embedding = avg_embedding.reshape(1, -1)
        idx = wrapper.predict(avg_embedding)
        print(f"Test prediction result: {idx}")
        

