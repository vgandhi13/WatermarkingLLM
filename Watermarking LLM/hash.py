import numpy as np
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans



class GloveSemanticHasher:
    def __init__(self, glove_path='glove.6B.300d.word2vec.txt', num_clusters=1000):
        try:
            self.glove = KeyedVectors.load_word2vec_format(glove_path, binary=False)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Converted GloVe embeddings not found. Ensure the conversion script has been run."
            )
        self.embedding_dim = self.glove.vector_size
        self.cache = {}
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.cluster_centers = None
        
    def get_embedding(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vectors = []
        
        for word in words:
            if word in self.cache:
                vector = self.cache[word]
            else:
                try:
                    vector = self.glove[word]
                    self.cache[word] = vector
                except KeyError:
                    continue
            vectors.append(vector)
                
        if not vectors:
            return np.zeros(self.embedding_dim)
            
        embedding = np.mean(vectors, axis=0)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def fit_clusters(self, texts):
        embeddings = [self.get_embedding(" ".join(gram)) for gram in texts]
        self.kmeans.fit(embeddings)
        self.cluster_centers = self.kmeans.cluster_centers_

        # Check for empty clusters
        labels, counts = np.unique(self.kmeans.labels_, return_counts=True)
        empty_clusters = set(range(self.kmeans.n_clusters)) - set(labels)
        if empty_clusters:
            print(f"Warning: The following clusters are empty: {empty_clusters}")

    def hash_3gram(self, tokens):
        if len(tokens) != 3:
            raise ValueError("Input must be exactly 3 tokens")
            
        text = " ".join(tokens)
        embedding = self.get_embedding(text)
        cluster_label = self.kmeans.predict([embedding])[0]
        return cluster_label

def create_3grams(text):
    words = text.split()
    return [words[i:i+3] for i in range(len(words)-2)]

def test_glove_hash():
    hasher = GloveSemanticHasher(num_clusters=1000)
    
    similar_3grams = [
        ["the", "cat", "sleeps"],
        ["the", "kitten", "rests"],
        ["a", "cat", "naps"],
    ]
    
    different_3grams = [
        ["the", "dog", "barks"],
        ["she", "runs", "fast"],
        ["they", "eat", "food"],
    ]
    
    # Fit clusters on a sample of 3-grams
    hasher.fit_clusters(similar_3grams + different_3grams)
    
    print("Similar 3-grams hash values:")
    for gram in similar_3grams:
        print(f"{gram}: {hasher.hash_3gram(gram)}")
        
    print("\nDifferent 3-grams hash values:")
    for gram in different_3grams:
        print(f"{gram}: {hasher.hash_3gram(gram)}")

if __name__ == "__main__":
    test_glove_hash()
