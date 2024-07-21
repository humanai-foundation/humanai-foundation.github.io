import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score
from gensim.models.coherencemodel import CoherenceModel
from bertopic import BERTopic
from llama_cpp import Llama
from gensim import corpora
from itertools import combinations
import numpy

def load_data_filtered(input_file: str, name_column: str) -> pd.DataFrame:
    """
    Load the filtered data from the input file.
    :param input_file: the input file containing the filtered data
    :param name_column: the name column
    :return: the filtered data
    """
    df = pd.read_csv(input_file)
    df = df.dropna(subset=[name_column])
    df = df.drop_duplicates(subset=[name_column], keep='first')
    print(df.shape[0])

    return df

def print_topics(topic_model: BERTopic, topics: list) -> None:
    """
    Print the topics and their words.
    :param topic_model: the BERTopic model
    :param topics: the list of topics
    """
    for topic_id in set(topics):
        print(f"Topic {topic_id}:")
        print(topic_model.get_topic(topic_id))

def assign_labels_to_topics(classifier, bert_model: BERTopic, zeroshot_topic_list: list, num_topics: int, 
                            threshold: float = 0.5) -> dict:
    """
    Assign labels to topics using zero-shot classification.
    :param classifier: the zero-shot classifier
    :param bert_model: the BERTopic model
    :param zeroshot_topic_list: the list of topics for zero-shot classification
    :param num_topics: the number of topics
    :param threshold: the score threshold for including labels
    :return: the dictionary of assigned labels for each topic
    """    
    topic_labels = {}
    
    topic_labels[-1] = 'outliers'

    for topic_idx in tqdm(range(num_topics), desc="Assigning labels to topics"):
        # Get the topic as a sequence of words
        topic_sequence = bert_model.get_topic(topic_idx)
        sequence_to_classify = " ".join([word for word, _ in topic_sequence])
        
        # Perform zero-shot classification
        res = classifier(sequence_to_classify, zeroshot_topic_list)
        
        # Get labels with scores above the threshold
        filtered_labels = [label for label, score in zip(res['labels'], res['scores']) if score >= threshold]

        # Determine topic name
        if not filtered_labels:
            filtered_labels = "-".join(set(bert_model.generate_topic_labels()[topic_idx + 1].split('_')[1:])).replace('-', ' - ')
        else:
            filtered_labels = " - ".join(filtered_labels)
        
        topic_labels[topic_idx] = filtered_labels

    return topic_labels

def save_assigned_labels(dict_labels: dict, output_file: str) -> None:
    """
    Save the assigned labels to a CSV file.
    :param dict_labels: the dictionary of assigned labels
    :param output_file: the output file
    """
    pd.DataFrame(list(dict_labels.items()), columns=['Topic', 'Labels']).to_csv(output_file, index=False)

def return_dataset(corpus: list, created_on: list, embeddings: numpy.ndarray, new_topics: list, 
                   probs: numpy.ndarray, topic_model: BERTopic, umap_embeddings: numpy.ndarray, 
                   save_umap_embeddings: bool = True) -> pd.DataFrame:
    """
    Filter and merge the data based on the identified indices.
    :param corpus: The original corpus.
    :param created_on: The original created_on dates.
    :param embeddings: The original embeddings.
    :param new_topics: The new topics.
    :param probs: The probabilities of the topics.
    :param topic_model: The BERTopic model.
    :param umap_embeddings: The UMAP embeddings.
    :param save_umap_embeddings: Whether to save the UMAP embeddings.
    :return: The filtered and merged DataFrame.
    """
    # Identify indices where the topic is not equal to -1
    indices = [index for index, topic in enumerate(new_topics) if topic != -1]

    # Filter the lists based on the identified indices
    corpus_valid = [corpus[i] for i in indices]
    created_on_valid = [created_on[i] for i in indices]
    embeddings_valid = [embeddings[i] for i in indices]
    topics_valid = [new_topics[i] for i in indices]
    probs_valid = [probs[i] for i in indices]

    # Create a DataFrame with the filtered data
    results = pd.DataFrame({
        'Document': corpus_valid,
        'Embedding': embeddings_valid,
        'Topic': topics_valid,
        'Probability': probs_valid,
        'Created_on': created_on_valid,
    })

    # Merge the results DataFrame with the topic information from the topic model
    results_final = pd.merge(results, topic_model.get_topic_info(), on='Topic')

    # Add umaap embeddings to the DataFrame
    if save_umap_embeddings:
        indices = [index for index, topic in enumerate(new_topics) if topic != -1]
        X=umap_embeddings[np.array(indices)]

        # Add UMAP embeddings to the DataFrame
        results_final['UMAP_embedding'] = list(X)

    return results_final

def calculate_dos(topic_words: dict, top_n: int = 10) -> float:
    """
    Calculate the average overlap score for all pairs of topics.
    :param topic_words: The topic words.
    :param top_n: The number of words to consider for each topic
    :return: The average overlap score.
    """
    overlap = 0
    num_combinations = 0
    for topic1, topic2 in combinations(topic_words.values(), 2):
        words1 = set([word for word, _ in topic1[:top_n]])
        words2 = set([word for word, _ in topic2[:top_n]])
        overlap += len(words1.intersection(words2))
        num_combinations += 1
        
    dos_score = overlap / num_combinations
    print(f"Distinta Overlap Score: {dos_score}")
    return dos_score

def calculate_silhouette_davies(umap_embeddings: numpy.ndarray, topics: list) -> tuple:
    """
    Evaluates topic clustering quality using silhouette and Davies-Bouldin scores.
    :param umap_embeddings: The UMAP embeddings of the data.
    :param topics: A list of topic assignments for each data point.
    :return: The silhouette and Davies-Bouldin scores.
    """
    # Get indices of topics that are not -1
    indices = [index for index, topic in enumerate(topics) if topic != -1]
    
    # Select the corresponding embeddings
    X = umap_embeddings[np.array(indices)]
    
    # Get the labels for the selected indices
    labels = [topic for _, topic in enumerate(topics) if topic != -1]
    
    # Calculate silhouette score
    silhouette_scores = silhouette_score(X, labels)
    
    # Calculate Davies-Bouldin score
    davies_bouldin_scores = davies_bouldin_score(X, labels)
    
    # Print the scores
    print(f"Silhouette Score: {silhouette_scores}")
    print(f"Davies-Bouldin Score: {davies_bouldin_scores}")

    return silhouette_scores, davies_bouldin_scores, X

def evaluate_topic_coherence(topic_words: dict, corpus: list, topn: int = 10, coherence_type: str = 'c_v') -> float:
    """
    Evaluates topic coherence using the CoherenceModel from gensim.
    :param topic_words: A list of topics with their corresponding words.
    :param corpus: A list of documents.
    :param topn: Number of top words to consider for each topic (default is 10).
    :param coherence_type: The type of coherence to calculate (default is 'c_v').
    :return: The coherence score of the model.
    """
    # Remove the last topic as it is the default topic
    topics_ll = [topic_words[i] for i in range(len(topic_words) - 1) if i != -1]
    # Extract the topn words for each topic
    topic_list = [[word for word, _ in topic[:topn]] for topic in topics_ll]
    
    # Split the documents into tokens
    texts = [doc.split() for doc in corpus]
    
    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(texts)
    
    # Create a CoherenceModel using the c_v metric
    coherence_model = CoherenceModel(
        topics=topic_list,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_type
    )
    
    # Calculate and return the coherence score
    coherence_score = coherence_model.get_coherence()
    print(f"Coherence Model: {coherence_score}")
    
    return coherence_score

def predict_topic(topic_model: BERTopic, sentence: list, num_classes: int = 5, custom_labels: bool = False) -> pd.DataFrame:
    """
    Predict the topic of a sentence using the BERTopic model.
    :param topic_model: The BERTopic model.
    :param sentence: The sentence to predict the topic of.
    :param num_classes: The number of classes to return.
    :param custom_labels: Whether to use custom labels.
    :return: A DataFrame with the predicted topics.
    """
    # Transform the sentence
    _, pr = topic_model.transform(sentence)

    # Get the top indices
    top_indices = np.argsort(pr[0])[::-1][:num_classes]

    # Get the top topics
    if custom_labels:
        top_topics = [(topic_model.get_topic(i), pr[0][i], topic_model.custom_labels_[i+1]) for i in top_indices]
    else:
        top_topics = [(topic_model.get_topic(i), pr[0][i], topic_model.generate_topic_labels()[i+1]) for i in top_indices]
    
    # Create a DataFrame with the results
    df_finals = pd.DataFrame(top_topics, columns=['Topic', 'Probability', 'Label'])

    # Extract the words and sentence
    df_finals['Words'] = df_finals['Topic'].apply(lambda topic: [word for word, _ in topic])

    df_finals['Sentence'] = sentence * len(df_finals)
    
    return df_finals

def generate_topic_label(llm, documents: list, keywords: str) -> str:
    """
    Generate a label for a topic based on the given documents and keywords using the LLM.
    :param llm: The LLM model to use for generation.
    :param documents: List of representative documents for the topic.
    :param keywords: Keywords describing the topic.
    :return: The generated label.
    """
    # Generate the prompt
    documents_str = "\n".join([f"- {doc}" for doc in documents])

    # Define the prompt template
    prompt_template = """ Q:
    I have a topic that contains the following documents:
    {documents}

    The topic is described by the following keywords: '{keywords}'.

    Based on the above information, can you give a short label of the topic of at most 5 words?
    A:
    """

    prompt = prompt_template.format(documents=documents_str, keywords=keywords)
    
    # Generate the label
    try:
        response = llm(prompt)
        return response['choices'][0]['text']
    except Exception as e:
        return "Error"

def process_dataset(llm, df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the given dataset to generate labels for each topic.
    :param llm: The LLM model to use for generation.
    :param df: The dataset to process.
    :return: A DataFrame containing the topic labels.
    """
    # Define the prompt template
    topic_labels = []

    # Process each row in the dataset
    for _, row in tqdm(df.iterrows(), desc="Processing dataset", total=len(df)):

        # Generate the topic label
        topic_id = row['Topic']
        documents = row['Representative_Docs']
        keywords = ", ".join(row['Representation'])
        label = generate_topic_label(llm, documents, keywords)

        # Append the topic label to the list
        topic_labels.append({
            'Topic': topic_id,
            'Label': label
        })

    topic_label = pd.DataFrame(topic_labels)

    # Remove leading numbers and periods
    topic_label['Label'] = topic_label['Label'].str.replace(r'\d+\.\s*', '', regex=True)
    
    # Remove leading dashes
    topic_label['Label'] = topic_label['Label'].str.replace(r'^\s*-\s*', '', regex=True)

    # Remove everything after and including colons
    topic_label['Label'] = topic_label['Label'].str.replace(r':.*', '', regex=True)

    # Remove everything after and including intermediate dashes 
    topic_label['Label'] = topic_label['Label'].str.replace(r'-.*', '', regex=True)

    # Remove isolated single characters
    topic_label['Label'] = topic_label['Label'].str.replace(r'\b\w\b', '', regex=True)

    # Replace multiple spaces with a single space
    topic_label['Label'] = topic_label['Label'].str.replace(r'\s+', ' ', regex=True)

    # Remove leading and trailing spaces
    topic_label['Label'] = topic_label['Label'].str.strip()

    # Remove double quotes
    topic_label['Label'] = topic_label['Label'].str.replace(r'"', '')

    # Remove single quotes
    topic_label['Label'] = topic_label['Label'].str.replace(r"'", '')

    # Remove everything after and including "Explanation" and replace commas with semicolons
    topic_label['Label'] = topic_label['Label'].str.replace(r'Explanation.*', '', regex=True)
    topic_label['Label'] = topic_label['Label'].str.replace(r',', ';')


    return topic_label