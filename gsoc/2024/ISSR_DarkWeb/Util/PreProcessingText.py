import nltk
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize, ne_chunk
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from gensim.models import LdaModel
from gensim.corpora import Dictionary
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_title(title: str) -> str:
    """
    Preprocesses text using tokenization, stopword removal, and lemmatization.
    :param title: the text to preprocess
    :return: the preprocessed text
    """
    # Tokenize and filter tokens
    tokens = [token for token in word_tokenize(title) if token.isalpha()]

    # Remove stopwords
    tokens = [token for token in tokens if token.lower() not in stopwords.words('english')]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(tokens)

def preprocess_content(content: str) -> str:
    """
    Preprocesses the content of a news article by tokenizing, lowercasing, and removing stopwords
    :param content: the content of a news article
    :return: the preprocessed content
    """
    # Tokenize and filter tokens
    tokens = [token.lower() for token in word_tokenize(content) if token.isalpha()]

    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

def clean_sentences(text: str) -> str:
    """
    Cleans a sentence by removing URLs, HTML tags, multiple spaces, punctuation, non-alphanumeric characters,
    leading and trailing spaces, and remaining underscores. 
    Separates numbers from words, keeps proper nouns capitalized, and ensures proper capitalization at the 
    beginning of sentences.
    :param text: the sentence to clean
    :return: the cleaned sentence
    """
    try:
        # Remove URLs
        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)

        # Replace / with space if it's between words
        text = re.sub(r'(?<=\w)/(?=\w)', ' ', text)

        # Remove leading and trailing spaces
        text = text.strip()

        # Remove HTML tags if present
        if '<' in text and '>' in text:
            text = BeautifulSoup(text, "html.parser").get_text()
            
        # Tokenize the text
        words = word_tokenize(text)

        # Tag parts of speech
        pos_tags = pos_tag(words)

        # Identify named entities
        named_entities = ne_chunk(pos_tags, binary=False)

        # Collect proper nouns
        proper_nouns = set()
        for subtree in named_entities:
            if isinstance(subtree, nltk.Tree):
                if subtree.label() in ['PERSON', 'ORGANIZATION', 'GPE', 'LOCATION']:
                    for leaf in subtree.leaves():
                        proper_nouns.add(leaf[0])

        # Convert to lowercase but keep proper nouns capitalized
        cleaned_words = []
        for word in words:
            if word in proper_nouns:
                cleaned_words.append(word)
            else:
                cleaned_words.append(word.lower())
        
        text = ' '.join(cleaned_words)

        # Remove punctuation (excluding spaces)
        text = re.sub(r'[^\w\s]', '', text)

        # Remove non-alphanumeric characters (excluding spaces and numbers)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Separate numbers from words by adding spaces around numbers
        text = re.sub(r'(\d+)', r' \1 ', text)

        # Remove any remaining underscores
        text = text.replace('_', '')

        # Remove single characters
        text = re.sub(r'\b\w\b', '', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Ensure the first letter of each sentence is correctly capitalized
        sentences = text.split('. ')
        try:
            cleaned_sentences = []
            for sentence in sentences:
                if sentence:
                    words = sentence.split()
                    # Convert the first word to lowercase unless it's a proper noun
                    if words[0] not in proper_nouns:
                        words[0] = words[0].lower()
                    cleaned_sentences.append(' '.join(words))
        except:
            pass
        text = '. '.join(cleaned_sentences).strip()
    except:
        return ''
    return text

def remove_single_characters(text: str) -> str:
    """
    Removes single characters from a text.
    :param text: the text to remove single characters from
    :return: the text with single characters removed
    """
    return re.sub(r'\b\w\b\s*', '', text)

def zero_shot_process_threads(df: pd.DataFrame, pipe, list_intents: list, output_file: str, 
                              label: str = 'name_thread') -> pd.DataFrame:
    """
    Process the threads in the DataFrame using the zero-shot classification pipeline.
    :param df: DataFrame containing the threads to process
    :param pipe: Zero-shot classification pipeline
    :param list_intents: List of intents to classify
    :param output_file: Path to the output CSV file
    :return: DataFrame with top 3 labels and scores for each thread
    """
    # Add columns for top labels and their respective scores
    for i in range(1, 4):
        df[f'top_label_{i}'] = None
        df[f'top_score_{i}'] = None

    # Dictionary to store already processed threads
    cache = {}
    
    # Extract unique name_thread values
    unique_threads = df[label].unique()

    for idx, thread_text in enumerate(tqdm(unique_threads, desc='Processing unique threads')):
        if thread_text not in cache:
            # Process new threads using the pipe function
            result = pipe(thread_text, list_intents)
            
            if result['labels']:
                sorted_labels = sorted(result['labels'], key=lambda x: result['scores'][result['labels'].index(x)], reverse=True)
                top_labels = sorted_labels[:3]
                top_scores = [result['scores'][result['labels'].index(label)] for label in top_labels]
            else:
                top_labels = [None, None, None]
                top_scores = [None, None, None]
            
            # Cache the results
            cache[thread_text] = (top_labels, top_scores)
        
        # Save to CSV every 10000 records
        if (idx + 1) % 10000 == 0:
            df_intermediate = df[df[label].isin(cache.keys())]
            df_intermediate.to_csv(f"{output_file}_{(idx + 1) // 10000}.csv", index=False)

    # Assign results to the appropriate columns
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Assigning results to DataFrame'):
        thread_text = str(row[label])
        if thread_text in cache:
            top_labels, top_scores = cache[thread_text]
            for i in range(3):
                df.at[index, f'top_label_{i+1}'] = top_labels[i]
                df.at[index, f'top_score_{i+1}'] = top_scores[i]

    # Final save to CSV
    df.to_csv(output_file, index=False)

    return df

def extract_top_keywords_tfidf(df: pd.DataFrame, num_keywords: int = 3) -> pd.DataFrame:
    """
    Extract the top keywords from the threads using TF-IDF.
    :param df: DataFrame containing the threads to process
    :param num_keywords: Number of top keywords to extract
    :return: DataFrame with top keywords for each thread
    """
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['name_thread'])

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Add columns for top keywords
    for i in range(1, num_keywords + 1):
        df[f'top_keyword_{i}'] = None

    # Extract top keywords for each thread
    for index in tqdm(range(len(df)), total=len(df), desc='Extracting top keywords'):
        tfidf_vector = tfidf_matrix[index]
        sorted_indices = tfidf_vector.toarray().argsort()[0][-num_keywords:][::-1]
        top_keywords = [feature_names[i] for i in sorted_indices]

        for i in range(num_keywords):
            df.at[index, f'top_keyword_{i+1}'] = top_keywords[i]

    return df

def compute_silhouette_score(lda_model: LdaModel, corpus: list) -> float:
    '''
    Compute the silhouette score for a given LDA model and corpus
    :param lda_model: the LDA model
    :param corpus: the corpus
    :return: the silhouette score
    '''
    # Compute the silhouette score
    topic_distributions = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
    topic_matrix = np.zeros((len(topic_distributions), lda_model.num_topics))
    for i, dist in enumerate(topic_distributions):
        for topic, prob in dist:
            topic_matrix[i, topic] = prob
    topic_matrix = normalize(topic_matrix, norm='l1', axis=1)
    score = silhouette_score(topic_matrix, np.argmax(topic_matrix, axis=1))
    return score

def save_best_score_to_csv(best_params: dict, best_score: float, output_file: str) -> None:
    '''
    Save the best parameters and score to a CSV file
    :param best_params: the best parameters
    :param best_score: the best score
    '''
    best_params['silhouette_score'] = best_score
    results_df = pd.DataFrame([best_params])
    results_df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)
    print(f"New best score saved to {output_file}")

def grid_search_lda(corpus: dict, dictionary: Dictionary, param_grid: dict, output_file: str) -> tuple:
    '''
    Perform a grid search over the specified parameters for LDA
    :param corpus: the corpus
    :param dictionary: the dictionary
    :param param_grid: the parameter grid
    :param output_file: the output file to save the results
    :return: the best model, best parameters, and best score
    '''
    best_score = -1
    best_params = None
    best_model = None
    print("Starting grid search")
    
    # Perform grid search
    for num_topics in param_grid['num_topics']:
        for passes in param_grid['passes']:
            print(f"Training LDA models with num_topics={num_topics} and passes={passes}")
            
            for iterations in param_grid['iterations']:

                # Train LDA model
                lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, 
                                    passes=passes, iterations=iterations, minimum_probability=0.01)
                
                # Compute silhouette score
                score = compute_silhouette_score(lda_model, corpus)

                # Update best score and parameters 
                if score > best_score:
                    best_score = score
                    best_params = {'num_topics': num_topics, 'passes': passes, 'iterations': iterations}
                    best_model = lda_model
                    save_best_score_to_csv(best_params, best_score, output_file)
                    print(f"Parameters: num_topics={num_topics}, passes={passes}, iterations={iterations}, silhouette score={score}")
                    
            print(f"Completed grid search iteration with num_topics={num_topics}, passes={passes}. Best silhouette score so far: {best_score}")
    
    print("Grid search completed")
    print(f"Best Parameters: {best_params}, Best Silhouette Score: {best_score}")
    
    return best_model, best_params, best_score