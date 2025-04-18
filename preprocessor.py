import re
from config import STOPWORDS

class TextPreprocessor:
    @staticmethod
    def preprocess_text(text):
        """
        Preprocess the input text by:
        1. Converting to lowercase
        2. Removing punctuation
        3. Removing stopwords
        4. Tokenizing
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            list: List of preprocessed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in STOPWORDS]
        
        # Debug: Print token statistics
        print(f"\nPreprocessing Statistics:")
        print(f"Original text length: {len(text)}")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Number of unique tokens: {len(set(tokens))}")
        print(f"Sample tokens: {tokens[:10]}")
        
        return tokens
    
    @classmethod
    def preprocess_documents(cls, documents):
        """
        Preprocess a list of documents
        
        Args:
            documents (list): List of text documents
            
        Returns:
            list: List of preprocessed document tokens
        """
        processed_docs = []
        for i, doc in enumerate(documents):
            print(f"\nProcessing document {i+1}/{len(documents)}")
            processed_docs.append(cls.preprocess_text(doc))
        return processed_docs
    
    @staticmethod
    def build_vocabulary(documents):
        """
        Build vocabulary from preprocessed documents
        
        Args:
            documents (list): List of preprocessed document tokens
            
        Returns:
            set: Set of unique words in the vocabulary
        """
        vocabulary = set()
        for doc in documents:
            vocabulary.update(doc)
            
        # Debug: Print vocabulary statistics
        print(f"\nVocabulary Statistics:")
        print(f"Total vocabulary size: {len(vocabulary)}")
        print(f"Sample vocabulary: {list(vocabulary)[:10]}")
        
        return vocabulary 