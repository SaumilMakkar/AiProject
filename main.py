import argparse
import os
from config import DATA_DIR, MODEL_DIR, CATEGORIES
from preprocessor import TextPreprocessor
from feature_extractor import FeatureExtractor
from naive_bayes import NaiveBayes as NaiveBayesClassifier
from utils import load_dataset, evaluate_model, save_model, load_model

def train_model(train_data_dir, model_save_dir):
    """
    Train the document classification model
    
    Args:
        train_data_dir (str): Path to training data directory
        model_save_dir (str): Path to save trained model
    """
    print("\n=== Training Model ===")
    print(f"Training data directory: {train_data_dir}")
    print(f"Model save directory: {model_save_dir}")
    print(f"Categories: {', '.join(CATEGORIES)}")
    
    print("\nLoading training data...")
    documents, labels, class_names = load_dataset(train_data_dir)
    
    print("\nPreprocessing documents...")
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)
    
    print("\nExtracting features...")
    feature_extractor = FeatureExtractor()
    feature_extractor.build_vocabulary(processed_docs)
    X = feature_extractor.extract_features_batch(processed_docs)
    
    print(f"\nVocabulary size: {len(feature_extractor.vocabulary)}")
    print(f"Number of training documents: {len(documents)}")
    
    print("\nTraining model...")
    model = NaiveBayesClassifier()
    model.fit(X, labels)
    
    print("\nSaving model...")
    save_model(model, preprocessor, feature_extractor, model_save_dir)
    
    print("\nTraining completed successfully!")

# def evaluate_test_data(test_data_dir, model_load_dir):
#     """
#     Evaluate model on test data
    
#     Args:
#         test_data_dir (str): Path to test data directory
#         model_load_dir (str): Path to load trained model
#     """
#     print("\n=== Evaluating Model ===")
#     print(f"Test data directory: {test_data_dir}")
#     print(f"Model load directory: {model_load_dir}")
    
#     print("\nLoading test data...")
#     documents, labels, class_names = load_dataset(test_data_dir)
    
#     print("\nLoading model...")
#     model, preprocessor, feature_extractor = load_model(model_load_dir)
    
#     print("\nPreprocessing test documents...")
#     processed_docs = preprocessor.preprocess_documents(documents)
#     X = feature_extractor.extract_features_batch(processed_docs)
    
#     print("\nMaking predictions...")
#     predictions = model.predict(X)
    
#     print("\nEvaluating model...")
#     results = evaluate_model(labels, predictions, class_names)
    
#     print("\n=== Evaluation Results ===")
#     print(f"Overall Accuracy: {results['accuracy']:.4f}")
#     print("\nPer-category Accuracy:")
#     for category, acc in results['category_accuracy'].items():
#         print(f"{category}: {acc:.4f}")
#     print("\nDetailed Classification Report:")
#     print(results['report'])

def evaluate_test_data(test_data_dir, model_load_dir):
    """
    Evaluate model on test data
    
    Args:
        test_data_dir (str): Path to test data directory
        model_load_dir (str): Path to load trained model
    """
    print("\n=== Evaluating Model ===")
    print(f"Test data directory: {test_data_dir}")
    print(f"Model load directory: {model_load_dir}")
    
    print("\nLoading test data...")
    documents, labels, class_names = load_dataset(test_data_dir)
    
    print("\nLoading model...")
    model, preprocessor, feature_extractor = load_model(model_load_dir)
    
    print("\nPreprocessing test documents...")
    processed_docs = preprocessor.preprocess_documents(documents)
    X = feature_extractor.extract_features_batch(processed_docs)
    
    print("\nMaking predictions...")
    # Get top-N predictions and extract only the top prediction for each sample
    top_n_labels, _ = model.predict(X)
    predictions = [labels[0] for labels in top_n_labels]  # Extract only the top prediction
    
    print("\nEvaluating model...")
    results = evaluate_model(labels, predictions, class_names)
    
    print("\n=== Evaluation Results ===")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print("\nPer-category Accuracy:")
    for category, acc in results['category_accuracy'].items():
        print(f"{category}: {acc:.4f}")
    print("\nDetailed Classification Report:")
    print(results['report'])

def classify_document(document_path, model_load_dir):
    """
    Classify a single document
    
    Args:
        document_path (str): Path to document to classify
        model_load_dir (str): Path to load trained model
    """
    print("\n=== Document Classification ===")
    print(f"Document path: {document_path}")
    print(f"Model load directory: {model_load_dir}")
    
    print("\nLoading document...")
    with open(document_path, 'r', encoding='utf-8') as f:
        document = f.read()
    
    print("\nLoading model...")
    model, preprocessor, feature_extractor = load_model(model_load_dir)
    
    print("\nPreprocessing document...")
    processed_doc = preprocessor.preprocess_text(document)
    X = feature_extractor.extract_features(processed_doc)
    
    print("\nMaking prediction...")
    prediction = model.predict([X])[0]
    probabilities = model.predict_proba([X])[0]  # NumPy array of probabilities
    
    print("\n=== Classification Results ===")
    print(f"Predicted category: {', '.join(prediction[0])}")


def main():
    parser = argparse.ArgumentParser(description='Document Classification System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--train_data_dir', type=str, default=DATA_DIR,
                            help='Path to training data directory')
    train_parser.add_argument('--model_save_dir', type=str, default=MODEL_DIR,
                            help='Path to save trained model')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--test_data_dir', type=str, default=DATA_DIR,
                           help='Path to test data directory')
    eval_parser.add_argument('--model_load_dir', type=str, default=MODEL_DIR,
                           help='Path to load trained model')
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify a document')
    classify_parser.add_argument('--document_path', type=str, required=True,
                               help='Path to document to classify')
    classify_parser.add_argument('--model_load_dir', type=str, default=MODEL_DIR,
                               help='Path to load trained model')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.train_data_dir, args.model_save_dir)
    elif args.command == 'evaluate':
        evaluate_test_data(args.test_data_dir, args.model_load_dir)
    elif args.command == 'classify':
        classify_document(args.document_path, args.model_load_dir)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 