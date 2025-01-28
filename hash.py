import numpy as np
from gensim.models import KeyedVectors



class GloveSemanticHasher:
    def __init__(self, glove_path='glove.6B.300d.word2vec.txt'):
        try:
            self.glove = KeyedVectors.load_word2vec_format(glove_path, binary=False)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Converted GloVe embeddings not found. Ensure the conversion script has been run."
            )
        self.embedding_dim = self.glove.vector_size
        self.cache = {}
        
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
    
    def hash_3gram(self, tokens, num_buckets=10000):
        if len(tokens) != 3:
            raise ValueError("Input must be exactly 3 tokens")
            
        text = " ".join(tokens)
        embedding = self.get_embedding(text)
        hash_value = int(sum(embedding[:4] * 1000) % num_buckets)
        return hash_value

def create_3grams(text):
    words = text.split()
    return [words[i:i+3] for i in range(len(words)-2)]

def test_glove_hash():
    hasher = GloveSemanticHasher()
    
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
    
    print("Similar 3-grams hash values:")
    for gram in similar_3grams:
        print(f"{gram}: {hasher.hash_3gram(gram)}")
        
    print("\nDifferent 3-grams hash values:")
    for gram in different_3grams:
        print(f"{gram}: {hasher.hash_3gram(gram)}")

if __name__ == "__main__":
    test_glove_hash()
