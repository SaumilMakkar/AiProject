from collections import Counter
from config import MIN_WORD_FREQUENCY, MAX_FEATURES

class FeatureExtractor:
    def __init__(self):
        self.vocabulary = None
        self.word_to_index = None
        
    def build_vocabulary(self, documents):
        """
        Build vocabulary from documents and create word to index mapping
        
        Args:
            documents (list): List of preprocessed document tokens
        """
        # Count word frequencies
        word_counts = Counter()
        for doc in documents:
            word_counts.update(doc)
            
        # Debug: Print word frequency statistics
        print("\nWord Frequency Statistics:")
        print(f"Total unique words: {len(word_counts)}")
        print(f"Most common words: {word_counts.most_common(10)}")
        print(f"Words with frequency < {MIN_WORD_FREQUENCY}: {sum(1 for count in word_counts.values() if count < MIN_WORD_FREQUENCY)}")
            
        # Filter words by frequency
        vocabulary = {word for word, count in word_counts.items() 
                     if count >= MIN_WORD_FREQUENCY}
        
        # Limit vocabulary size
        if len(vocabulary) > MAX_FEATURES:
            vocabulary = set(sorted(vocabulary, key=lambda x: word_counts[x], 
                                 reverse=True)[:MAX_FEATURES])
            
        self.vocabulary = vocabulary
        self.word_to_index = {word: int(idx) for idx, word in enumerate(sorted(vocabulary))}
        
        # Debug: Print vocabulary statistics
        print("\nFinal Vocabulary Statistics:")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Sample word-to-index mapping: {list(self.word_to_index.items())[:10]}")
        
    def extract_features(self, document):
        """
        Convert document to feature vector using Bag of Words
        
        Args:
            document (list): Preprocessed document tokens
            
        Returns:
            list: Feature vector
        """
        if not self.vocabulary:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
            
        # Initialize feature vector
        features = [0] * len(self.vocabulary)
        
        # Count word occurrences
        word_counts = Counter(document)
        for word, count in word_counts.items():
            if word in self.word_to_index:
                idx = int(self.word_to_index[word])
                features[idx] = count
                
        # Debug: Print feature vector statistics
        print("\nFeature Vector Statistics:")
        print(f"Document length: {len(document)}")
        print(f"Non-zero features: {sum(1 for x in features if x > 0)}")
        print(f"Feature vector sum: {sum(features)}")
        
        return features
    
    def extract_features_batch(self, documents):
        """
        Convert multiple documents to feature vectors
        
        Args:
            documents (list): List of preprocessed document tokens
            
        Returns:
            list: List of feature vectors
        """
        feature_vectors = []
        for i, doc in enumerate(documents):
            print(f"\nExtracting features for document {i+1}/{len(documents)}")
            feature_vectors.append(self.extract_features(doc))
        return feature_vectors 