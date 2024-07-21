import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA, KernelPCA
import openTSNE
import umap.umap_ as umap
import hdbscan


class TextClustering:
    """
    Class for clustering text data using SentenceTransformer and UMAP or t-SNE.
    """
    def __init__(self, data_filtered: pd.DataFrame, text_column: str) -> None:
        """
        Initialize the TextClustering class.
        :param data_filtered: the filtered data
        :param text_column: the column containing the text data
        """
        self.data_filtered = data_filtered
        self.text_column = text_column
        self.corpus = self.load_corpus()
        self.corpus_embeddings = None
        self.directory_embeddings = None
        self.directory_embedding_previous = None
        self.logger = self.initialize_logger()

    def initialize_logger(self) -> logging.Logger:
        """
        Initialize the logger for the class.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Check if the logger already has handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def load_corpus(self) -> list:
        """
        Load and preprocess the corpus.
        :return: the preprocessed corpus
        """
        corpus = self.data_filtered[self.text_column].tolist()
        corpus = [x.lower() for x in corpus]
        corpus = list(set(corpus))
        return corpus

    def encode_corpus(self, model: SentenceTransformer, batch_size: int, to_tensor: bool) -> np.ndarray:
        """
        Compute embeddings for the sentences.
        :param model: the SentenceTransformer model to use
        :param batch_size: the batch size for encoding
        :return: the embeddings of the corpus
        """
        self.logger.info("Encoding the corpus. This might take a while.")
        self.corpus_embeddings = model.encode(self.corpus, batch_size=batch_size, convert_to_tensor=to_tensor, show_progress_bar=True)
        return self.corpus_embeddings

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length.
        :param embeddings: the embeddings to normalize
        :return: the normalized embeddings
        """
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def reduce_dimensionality(self, n_neighbors: int = 15, n_components: int = 10) -> np.ndarray:
        """
        Reduce dimensionality of embeddings with UMAP.
        :param n_neighbors: the number of neighbors to consider
        :param n_components: the number of components to reduce to
        :return: the reduced embeddings
        """
        self.logger.info("Performing dimensionality reduction with UMAP")
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine')
        return reducer.fit_transform(self.corpus_embeddings)

    def reduce_dimensionality_UMAP(self, batch_size: int, n_neighbors: int = 15, n_components: int = 10) -> np.ndarray:
        """
        Reduce dimensionality of embeddings with UMAP.
        :param n_neighbors: the number of neighbors to consider
        :param n_components: the number of components to reduce to
        :param batch_size: the size of each batch for transformation
        :return: the reduced embeddings
        """
        self.logger.info("Performing dimensionality reduction with UMAP")
        
        total_samples = self.corpus_embeddings.shape[0]
        
        # Fit UMAP on the entire dataset
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine')
        self.logger.info("Fitting UMAP on the entire dataset")
        reducer.fit(self.corpus_embeddings)
        
        # Transform data in batches
        all_reduced_embeddings = []
        num_batches = int(np.ceil(total_samples / batch_size))
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            batch_embeddings = self.corpus_embeddings[start_idx:end_idx]

            self.logger.info(f"Transforming batch {batch_idx + 1}/{num_batches}")
            reduced_batch_embeddings = reducer.transform(batch_embeddings)
            all_reduced_embeddings.append(reduced_batch_embeddings)

        reduced_embeddings = np.vstack(all_reduced_embeddings)
        return reduced_embeddings

    def reduce_dimensionality_tsne(self, n_components: int = 2, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
        """
        Reduce dimensionality of embeddings with t-SNE.
        :param n_components: the number of components to reduce to
        :param perplexity: the perplexity value
        :param n_iter: the number of iterations
        :return: the reduced embeddings
        """
        self.logger.info("Performing dimensionality reduction with t-SNE")
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, metric='cosine')
        reduced_embeddings = tsne.fit_transform(self.corpus_embeddings)
        return reduced_embeddings.astype(np.float32)

    def reduce_dimensionality_Opentsne(self, n_components: int = 2, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
        """
        Reduce dimensionality of embeddings with t-SNE using openTSNE.
        :param n_components: the number of components to reduce to
        :param perplexity: the perplexity value
        :param n_iter: the number of iterations
        :return: the reduced embeddings
        """
        self.logger.info("Performing dimensionality reduction with incremental t-SNE")
        embeddings_np = self.corpus_embeddings.cpu().numpy()
        tsne = openTSNE.TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            metric="cosine",
            initialization="pca",
            negative_gradient_method="fft",
            n_jobs=-1
        )
        reduced_embeddings = tsne.fit(embeddings_np)
        return np.array(reduced_embeddings, dtype=np.float32)

    def determine_optimal_components(self, threshold: float = 0.9) -> int:
        """
        Determine the optimal number of components for PCA based on explained variance threshold.
        :param threshold: the cumulative explained variance threshold to reach
        :return: the optimal number of components
        """
        self.logger.info("Determining the optimal number of components using PCA")

        pca = PCA()
        pca.fit(self.corpus_embeddings)

        explained_variance_ratio = pca.explained_variance_ratio_
        cum_explained_variance = np.cumsum(explained_variance_ratio)

        # Plot cumulative explained variance
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(cum_explained_variance) + 1), cum_explained_variance, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Optimal Number of Components')
        plt.grid(True)
        plt.show()

        optimal_components = np.argmax(cum_explained_variance >= threshold) + 1
        self.logger.info(f"Optimal number of components: {optimal_components}")
        self.logger.info(f"Cumulative explained variance reached: {cum_explained_variance[optimal_components - 1]}")

        return optimal_components

    def reduce_dimensionality_PCA(self, n_components: int) -> np.ndarray:
        """
        Reduce dimensionality of embeddings with PCA.
        :param n_components: the number of components to reduce to
        :return: the reduced embeddings
        """
        self.logger.info(f"Performing dimensionality reduction with PCA to {n_components} components")
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(self.corpus_embeddings)
        return reduced_embeddings

    def reduce_dimensionality_KPCA(self, n_components: int, kernel: str = 'rbf', gamma: float = 0.1) -> np.ndarray:
        """
        Reduce dimensionality of embeddings with Kernel PCA.
        :param n_components: the number of components to reduce to
        :param kernel: the kernel function to use ('rbf', 'linear', 'poly', 'sigmoid', etc.)
        :param gamma: kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        :return: the reduced embeddings
        """
        self.logger.info(f"Performing dimensionality reduction with Kernel PCA to {n_components} components")

        kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
        reduced_embeddings = kpca.fit_transform(self.corpus_embeddings)
        
        return reduced_embeddings

    def reduce_dimensionalityPCA_UMAP(self, n_neighbors: int = 15, n_components: int = 10, 
                                      pca_components: int = 50) -> np.ndarray:
        """
        Reduce dimensionality of embeddings with UMAP, optionally using PCA for preliminary reduction.
        :param n_neighbors: the number of neighbors to consider
        :param n_components: the number of components to reduce to
        :param pca_components: the number of components for PCA preliminary reduction
        :return: the reduced embeddings
        """
        self.logger.info("Performing preliminary dimensionality reduction with PCA")
        pca = PCA(n_components=pca_components)
        reduced_data = pca.fit_transform(self.corpus_embeddings)
        
        self.logger.info("Performing dimensionality reduction with UMAP")
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine')
        return reducer.fit_transform(reduced_data)

    def perform_community_detection(self, min_community_size: int = 10, threshold: float = 0.7) -> dict:
        """
        Perform clustering with community_detection.
        :param min_community_size: the minimum size of a community
        :param threshold: the threshold to consider for community detection
        :return: the clusters
        """
        self.logger.info("Starting clustering with community_detection")
        clusters = util.community_detection(self.corpus_embeddings, min_community_size=min_community_size, threshold=threshold)
        return clusters

    def perform_agglomerative_clustering(self, cluster_embeddings: np.ndarray, n_clusters: int, 
                                         metric: str = 'cosine', linkage: str = 'average') -> tuple:
        """
        Perform Agglomerative Clustering on embeddings.
        :param cluster_embeddings: the embeddings of the clusters
        :param n_clusters: the number of clusters
        :return: the clusters and the cluster assignment
        """
        self.logger.info("Starting clustering with Agglomerative Clustering")
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
        clustering_model.fit(cluster_embeddings)
        cluster_assignment = clustering_model.labels_
        clusters = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(sentence_id)
        return clusters, cluster_assignment

    def perform_agglomerative_clustering_in_batches_prev(self, cluster_embeddings: np.ndarray, batch_size: int, 
                                                         n_clusters: int, metric: str = 'cosine', 
                                                         linkage: str = 'average') -> tuple:
        """
        Perform Agglomerative Clustering on embeddings in batches.
        :param cluster_embeddings: the embeddings of the clusters
        :param n_clusters: the number of clusters
        :param batch_size: the size of each batch
        :return: the clusters and the cluster assignment
        """
        self.logger.info("Starting clustering with Agglomerative Clustering in batches")
        total_samples = cluster_embeddings.shape[0]
        num_batches = int(np.ceil(total_samples / batch_size))
        
        all_cluster_assignments = np.empty(total_samples, dtype=int)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            batch_embeddings = cluster_embeddings[start_idx:end_idx]

            clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
            cluster_assignment = clustering_model.fit_predict(batch_embeddings)
            
            all_cluster_assignments[start_idx:end_idx] = cluster_assignment + batch_idx * n_clusters

        # Correct cluster IDs to be consistent across batches
        unique_labels = np.unique(all_cluster_assignments)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        final_cluster_assignment = np.vectorize(label_map.get)(all_cluster_assignments)

        clusters = {}
        for sentence_id, cluster_id in enumerate(final_cluster_assignment):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(sentence_id)

        return clusters, final_cluster_assignment
    
    def perform_agglomerative_clustering_in_batches(self, cluster_embeddings: np.ndarray, batch_size: int, 
                                                    n_clusters: int, metric: str, linkage: str) -> tuple:
        """
        Perform Agglomerative Clustering on embeddings in batches.
        :param cluster_embeddings: the embeddings of the clusters
        :param n_clusters: the number of clusters
        :param batch_size: the size of each batch
        :return: the clusters and the cluster assignment
        """
        self.logger.info("Starting clustering with Agglomerative Clustering in batches")
        total_samples = cluster_embeddings.shape[0]
        num_batches = int(np.ceil(total_samples / batch_size))
        
        all_clusters = []
        all_cluster_assignments = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            batch_embeddings = cluster_embeddings[start_idx:end_idx]

            clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
            clustering_model.fit(batch_embeddings)
            cluster_assignment = clustering_model.labels_
            
            batch_clusters = {}
            for sentence_id, cluster_id in enumerate(cluster_assignment):
                if cluster_id not in batch_clusters:
                    batch_clusters[cluster_id] = []
                batch_clusters[cluster_id].append(start_idx + sentence_id)
            
            all_clusters.append(batch_clusters)
            all_cluster_assignments.append(cluster_assignment)

        # Merge clusters from all batches
        merged_clusters = {}
        for batch_clusters in all_clusters:
            for cluster_id, sentence_ids in batch_clusters.items():
                if cluster_id not in merged_clusters:
                    merged_clusters[cluster_id] = []
                merged_clusters[cluster_id].extend(sentence_ids)

        # Create final cluster assignment array
        final_cluster_assignment = np.concatenate(all_cluster_assignments)

        return merged_clusters, final_cluster_assignment

    def k_means(self, cluster_embeddings: np.ndarray, batch_size: int, n_clusters: int = 10) -> tuple:
        """
        Perform MiniBatchKMeans clustering on embeddings in batches.
        :param cluster_embeddings: the embeddings of the clusters
        :param n_clusters: the number of clusters
        :param batch_size: the size of each batch
        :return: the clusters and the cluster assignment
        """
        self.logger.info("Starting clustering with MiniBatch K-Means Clustering")
        
        # Initialize MiniBatchKMeans clustering
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
        cluster_assignment = kmeans.fit_predict(cluster_embeddings)

        # Prepare clusters dictionary
        clusters = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(sentence_id)

        return clusters, cluster_assignment

    def plot_elbow_method(self, cluster_embeddings: np.ndarray, batch_size: int, max_k: int = 25) -> None:
        """
        Plot the elbow method for determining the optimal number of clusters.
        :param cluster_embeddings: the embeddings of the clusters
        :param batch_size: the size of each batch
        :param max_k: the maximum number of clusters to test
        """
        self.logger.info("Plotting the Elbow Method for K-Means")
        wcss = []
        k_values = range(1, max_k + 1)
        
        for k in tqdm(k_values, desc="Evaluating K-Means Clustering"):
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42)
            kmeans.fit(cluster_embeddings)
            wcss.append(kmeans.inertia_)  # Inertia is the sum of squared distances to the closest cluster center
        
        plt.figure(figsize=(10, 8))
        plt.plot(k_values, wcss, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    def perform_hdbscan_clustering_batches(self, cluster_embeddings: np.ndarray, min_cluster_size: float, 
                                           min_samples: int, batch_size: int) -> tuple:
        """
        Perform HDBSCAN on embeddings in batches.
        :param cluster_embeddings: the embeddings of the clusters
        :param min_cluster_size: the minimum cluster size
        :param min_samples: the minimum number of samples
        :param batch_size: the size of each batch
        :return: the clusters and the cluster assignment
        """
        self.logger.info("Starting clustering with HDBSCAN in batches")
        total_samples = cluster_embeddings.shape[0]
        num_batches = int(np.ceil(total_samples / batch_size))
        
        all_clusters = []
        all_cluster_assignments = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            batch_embeddings = cluster_embeddings[start_idx:end_idx]

            clustering_model = hdbscan.HDBSCAN(min_cluster_size=max(1, int(len(self.data_filtered) * min_cluster_size)), min_samples=min_samples)
            clustering_model.fit(batch_embeddings)
            cluster_assignment = clustering_model.labels_
            
            batch_clusters = {}
            for sentence_id, cluster_id in enumerate(cluster_assignment):
                if cluster_id == -1:
                    # Ignore noise points
                    continue
                if cluster_id not in batch_clusters:
                    batch_clusters[cluster_id] = []
                batch_clusters[cluster_id].append(start_idx + sentence_id)
            
            all_clusters.append(batch_clusters)
            all_cluster_assignments.append(cluster_assignment)

        # Merge clusters from all batches
        merged_clusters = {}
        for batch_clusters in all_clusters:
            for cluster_id, sentence_ids in batch_clusters.items():
                if cluster_id not in merged_clusters:
                    merged_clusters[cluster_id] = []
                merged_clusters[cluster_id].extend(sentence_ids)

        # Create final cluster assignment array
        final_cluster_assignment = np.concatenate(all_cluster_assignments)

        return merged_clusters, final_cluster_assignment

    def perform_hdbscan_clustering(self, cluster_embeddings: np.ndarray, min_cluster_size: float, 
                                   min_samples: int) -> tuple:
        """
        Perform HDBSCAN clustering on embeddings.
        :param cluster_embeddings: the embeddings of the clusters
        :param min_cluster_size: the minimum cluster size
        :param min_samples: the minimum number of samples
        :return: the clusters and the cluster assignment
        """
        self.logger.info("Starting clustering with HDBSCAN")
        clustering_model = hdbscan.HDBSCAN(min_cluster_size=max(1, int(len(self.data_filtered) * min_cluster_size)), min_samples=min_samples)
        clustering_model.fit(cluster_embeddings)
        cluster_assignment = clustering_model.labels_
        clusters = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id == -1:
                # Ignore noise points
                continue
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(sentence_id)
        return clusters, cluster_assignment
        
    def plot_clusters(self, embeddings_2d: np.ndarray, cluster_labels: np.ndarray) -> None:
        """
        Visualize clusters in 2D as dots.
        :param embeddings_2d: the 2D embeddings
        :param cluster_labels: the labels of the clusters
        """
        plt.figure(figsize=(10, 8))
        unique_labels = set(cluster_labels)
        # Generate a different color for each cluster
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Color for noise (cluster -1)
                col = 'k'
                marker = 'x'
            else:
                marker = 'o'
            class_member_mask = (cluster_labels == k)
            xy = embeddings_2d[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], marker, markerfacecolor=col, markeredgecolor='k', markersize=6, alpha=0.6, label=f'Cluster {k}')
        plt.title('2D Cluster Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.show()

    def show_name_clusters(self, cluster: list, kw_model) -> tuple:
        """
        Show the names of the clusters.
        :param cluster: the cluster to show the names of
        :param kw_model: the KeyBERT model to use
        :return: the top and bottom words of the clusters
        """
        used = []
        doc = ""
        for sentence_id in cluster:
            doc += str(self.corpus[sentence_id]) + " "
            used.append(sentence_id)
        self.logger.info("Extracting top 5 keywords from cluster")
        n_1_topics_ = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=None)
        self.logger.info(f"1-gram topics: {n_1_topics_}")
        n_2_topics_ = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=None)
        self.logger.info(f"1-2-gram topics: {n_2_topics_}")
        return n_1_topics_, n_2_topics_, used

    def summarize(self, clusters: dict, n_words: int, name_model: str) -> pd.DataFrame:
        """
        Summarize the clusters with top and bottom words and include cluster size percentage.
        :param clusters: the clusters to summarize
        :param n_words: the number of words to show
        :param name_model: the name of the SentenceTransformer model
        :return: the summarized clusters
        """
        results = {}
        self.logger.info("Extracting keywords from final clusters")
        total_elements = sum(len(cluster) for cluster in clusters.values())
        # Initialize KeyBERT model
        kw_model = KeyBERT(model=name_model)
        for cluster_id, cluster in tqdm(clusters.items(), desc="Processing clusters"):
            # Show the name of the cluster
            self.logger.info("\nCluster {}, #{} Elements".format(cluster_id, len(cluster)))
            n_1_topics_, n_2_topics_, used = self.show_name_clusters(cluster, kw_model)
            
            cluster_size_percentage = (len(cluster) / total_elements) * 100
            # Show the top and bottom words of the cluster
            results[cluster_id] = {
                'size_percentage': cluster_size_percentage,
                'n_1_topics': n_1_topics_,
                'n_2_topics': n_2_topics_
            }
            
            for i, sentence_id in enumerate(cluster[:n_words]):
                try:
                    sentence = self.corpus[sentence_id]
                    results[cluster_id][f'top_{i}'] = sentence
                except Exception as e:
                    self.logger.error(f"Error processing top_{i} of cluster {cluster_id}: {e}")
                    continue
            
            for i, sentence_id in enumerate(cluster[-n_words:]):
                try:
                    sentence = self.corpus[sentence_id]
                    results[cluster_id][f'bottom_{i}'] = sentence
                except Exception as e:
                    self.logger.error(f"Error processing bottom_{i} of cluster {cluster_id}: {e}")
                    continue

        return pd.DataFrame(results).T
    
    def evaluate_clusters(self, filtered_embeddings: np.ndarray, filtered_labels: np.ndarray) -> None:
        """
        Evaluate the clusters using Silhouette Score and Davies-Bouldin Index.
        :param filtered_embeddings: the filtered embeddings
        :param filtered_labels: the filtered labels
        """
        self.logger.info("Evaluating Silhouette Score")
        sil_score = silhouette_score(filtered_embeddings, filtered_labels)
        self.logger.info(f"Silhouette Score: {sil_score}")
        self.logger.info("Evaluating Davies-Bouldin Index")
        db_score = davies_bouldin_score(filtered_embeddings, filtered_labels)
        self.logger.info(f"Davies-Bouldin Index: {db_score}")

    def save_embeddings(self, directory: str, cluster_assignment: np.ndarray, non_outlier_indices: np.ndarray, 
                        filtered_embeddings: np.ndarray) -> None:
        """
        Save the embeddings with indices.
        :param cluster_assignment: the cluster assignment
        :param non_outlier_indices: the indices of non-outliers
        :param filtered_embeddings: the filtered embeddings
        """
        embeddings_with_indices = {
            'embeddings': filtered_embeddings,
            'indices': np.arange(len(cluster_assignment))[non_outlier_indices]
        }
        np.save(directory, embeddings_with_indices)

    def load_embeddings(self, to_frame: bool = False) -> pd.DataFrame:
        """
        Load the embeddings with indices.
        :param to_frame: whether to convert to a DataFrame
        :return: the embeddings with indices
        """
        embeddings_with_indices = np.load(f"{self.directory_embeddings}.npy", allow_pickle=True).item()
        embeddings_with_indices_previous = np.load(f"{self.directory_embedding_previous}.npy", allow_pickle=True).item()
        if to_frame:
            return pd.DataFrame({'index': embeddings_with_indices['embeddings'], 'embedding': embeddings_with_indices['indices']}), pd.DataFrame({'index': embeddings_with_indices_previous['embeddings'], 'embedding': embeddings_with_indices_previous['indices']})
        else:
            return embeddings_with_indices['embeddings'], embeddings_with_indices['indices'], embeddings_with_indices_previous['embeddings'], embeddings_with_indices_previous['indices']


    def main(self, name_model: str, to_tensor: bool = True, reduce_previous: bool = False, save_embeddings: bool = False, 
             directory_embeddings: str = None, batch_size: int = 32, batch_cluster_size: int = 33370, 
             exec_reduction: bool = True, reduction: str = 'tSNE', n_neighbors: int = 15, n_components_umap: int = 10, 
             n_words: int = 20, n_components: int = 2, perplexity: int = 30, n_iter: int = 1000, n_clusters: int = 6, 
             hdbscan: bool = False, hdbscan_batches: bool = True, agglomerative_batches: bool = True, k_means: bool = False, 
             elbow: bool = False, min_cluster_size: float = 0.02, min_samples: int = None, threshold: float = 0.8) -> tuple:
        """
        Main function to cluster text data.
        :param name_model: the name of the SentenceTransformer model to use
        :param reduce_previous: whether to reduce the previous embeddings
        :param save_embeddings: whether to save the embeddings
        :param directory_embeddings: the directory to save the embeddings
        :param batch_size: the batch size for encoding
        :param exec_reduction: whether to execute dimensionality reduction
        :param reduction: the reduction technique to use
        :param n_neighbors: the number of neighbors to consider for UMAP
        :param n_components_umap: the number of components to reduce to with UMAP
        :param n_words: the number of words to show in the summary
        :param n_components: the number of components to reduce to with t-SNE
        :param perplexity: the perplexity value for t-SNE
        :param n_iter: the number of iterations for t-SNE
        :param n_clusters: the number of clusters to create
        :param hdbscan: whether to use HDBSCAN for clustering
        :param hdbscan_batches: whether to use HDBSCAN in batches
        :param agglomerative_batches: whether to use Agglomerative Clustering in batches
        :param k_means: whether to use K-Means for clustering
        :param elbow: whether to plot the elbow method for K-Means
        :param min_cluster_size: the minimum cluster size for HDBSCAN
        :param min_samples: the minimum number of samples for HDBSCAN
        :param threshold: the threshold for PCA
        :return: the summarized clusters
        """
        model = SentenceTransformer(name_model)

        # Encode the corpus
        self.encode_corpus(model, batch_size=batch_size, to_tensor=to_tensor)

        # Normalize embeddings prior to dimensionality reduction
        if reduce_previous:
            self.corpus_embeddings = self.normalize_embeddings(self.corpus_embeddings)
            processed_embeddings = self.corpus_embeddings
        else:
            processed_embeddings = self.normalize_embeddings(self.corpus_embeddings)


        # Reduce dimensionality
        distance = 'euclidean'
        linkage='ward'
        if exec_reduction:
            if reduction == 'UMAP':
                red_embeddings = self.reduce_dimensionality_UMAP(n_neighbors=n_neighbors, n_components=n_components_umap, batch_size=batch_cluster_size)
            elif (reduction == 'PCA' or reduction == 'KPCA'):
                optimal_components = self.determine_optimal_components(threshold=threshold)
                if reduction == 'PCA':
                    red_embeddings = self.reduce_dimensionality_PCA(n_components=optimal_components)
                else:
                    red_embeddings = self.reduce_dimensionality_KPCA(n_components=optimal_components)
            elif reduction == 'PCAUMAP':
                red_embeddings = self.reduce_dimensionalityPCA_UMAP(n_neighbors=n_neighbors, n_components=n_components, pca_components=50)
            else:
                red_embeddings = self.reduce_dimensionality_Opentsne(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
        else:
            distance = 'cosine'
            linkage="average"
            red_embeddings = self.corpus_embeddings


        # Normalize embeddings
        self.corpus_embeddings = self.normalize_embeddings(red_embeddings)

        # Clustering
        if hdbscan:
            if hdbscan_batches:
                clusters, cluster_assignment = self.perform_hdbscan_clustering_batches(self.corpus_embeddings, min_cluster_size=min_cluster_size, min_samples=min_samples, batch_size=batch_cluster_size)
            else:
                clusters, cluster_assignment = self.perform_hdbscan_clustering(self.corpus_embeddings, min_cluster_size=min_cluster_size, min_samples=min_samples)
        elif k_means:
            clusters, cluster_assignment = self.k_means(self.corpus_embeddings, batch_size=batch_cluster_size, n_clusters=n_clusters)
            if elbow:
                self.plot_elbow_method(self.corpus_embeddings, batch_size, max_k=25)
        else:
            if agglomerative_batches:
                clusters, cluster_assignment = self.perform_agglomerative_clustering_in_batches(self.corpus_embeddings, n_clusters=n_clusters, batch_size=batch_cluster_size, metric=distance, linkage=linkage)
            else:
                clusters, cluster_assignment = self.perform_agglomerative_clustering(self.corpus_embeddings, n_clusters=n_clusters, metric=distance, linkage=linkage)

        # Plotting clusters
        if exec_reduction:
            self.plot_clusters(red_embeddings, cluster_assignment)
        else:
            red_embeddings = self.reduce_dimensionality_Opentsne(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
            self.plot_clusters(red_embeddings, cluster_assignment)
        
        # Associating clusters with original DataFrame entries
        cluster_df = self.data_filtered.copy()
        cluster_df['cluster'] = cluster_assignment

        # Filter out outliers (label -1)
        non_outlier_indices = np.where(cluster_assignment != -1)
        filtered_embeddings = self.corpus_embeddings[non_outlier_indices]
        filtered_labels = cluster_assignment[non_outlier_indices]

        # Save embeddings with indices
        if save_embeddings:
            self.directory_embeddings = directory_embeddings
            self.save_embeddings(self.directory_embeddings, cluster_assignment, non_outlier_indices, filtered_embeddings)
            self.directory_embedding_previous = f"{directory_embeddings}_previous"
            self.save_embeddings(self.directory_embedding_previous, cluster_assignment, non_outlier_indices, processed_embeddings)

        # Evaluate clusters
        self.evaluate_clusters(filtered_embeddings, filtered_labels)

        # Summarize clusters
        summarized_clusters = self.summarize(clusters, n_words, name_model).reset_index()

        # Merge cluster information with original DataFrame
        result_final = pd.merge(cluster_df, summarized_clusters, left_on='cluster', right_on='index')

        return summarized_clusters, cluster_df, result_final
