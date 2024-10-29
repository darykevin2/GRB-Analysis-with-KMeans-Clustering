import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import os

def load_and_preprocess_data(filepath):
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Convert T90 and fluence to log scale
    df['log_t90'] = np.log10(df['t90'] + 1e-6)
    df['log_fluence'] = np.log10(df['fluence'] + 1e-10)
    
    return df

def handle_missing_values(df):
    # Drop rows with NaN values
    df = df.dropna()
    return df

def perform_kmeans(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    return kmeans, clusters

def plot_and_save_elbow(X, feature_name, max_clusters=10):
    inertias = []
    silhouette_scores = []
    K = range(2, max_clusters+1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method For Optimal k ({feature_name})')
    
    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Score For Optimal k ({feature_name})')
    
    plt.tight_layout()
    plt.savefig(f'elbow_plot_{feature_name}.png')
    plt.close()
    
    return silhouette_scores

def plot_clustering_results(df, clusters, feature_type):
    plt.figure(figsize=(15, 10))
    
    if feature_type == "T90":
        # T90 distribution
        plt.subplot(2, 1, 1)
        for i in range(len(np.unique(clusters))):
            plt.hist(df['log_t90'][clusters == i], 
                    bins=30, alpha=0.5, 
                    label=f'Cluster {i}')
        plt.xlabel('log T90 (s)')
        plt.ylabel('Count')
        plt.title('T90 Distribution by Cluster', fontsize=16, fontweight='bold')
        plt.legend()
        
        # T90 vs Fluence scatter
        plt.subplot(2, 1, 2)
        scatter = plt.scatter(df['log_t90'], df['log_fluence'], 
                            c=clusters, cmap='viridis', alpha=0.6)
        plt.xlabel('log T90 (s)')
        plt.ylabel('log Fluence (erg/cm²)')
        plt.title('GRB Clusters based on T90', fontsize=16, fontweight='bold')
        plt.colorbar(scatter)
        
    elif feature_type == "Fluence":
        # Fluence distribution
        plt.subplot(2, 1, 1)
        for i in range(len(np.unique(clusters))):
            plt.hist(df['log_fluence'][clusters == i], 
                    bins=30, alpha=0.5, 
                    label=f'Cluster {i}')
        plt.xlabel('log Fluence (erg/cm²)')
        plt.ylabel('Count')
        plt.title('Fluence Distribution by Cluster', fontsize=16, fontweight='bold')
        plt.legend()
        
        # T90 vs Fluence scatter
        plt.subplot(2, 1, 2)
        scatter = plt.scatter(df['log_t90'], df['log_fluence'], 
                            c=clusters, cmap='viridis', alpha=0.6)
        plt.xlabel('log T90 (s)')
        plt.ylabel('log Fluence (erg/cm²)')
        plt.title('GRB Clusters based on Fluence', fontsize=16, fontweight='bold')
        plt.colorbar(scatter)
        
    else:  # Both features
        # T90 vs Fluence scatter
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(df['log_t90'], df['log_fluence'], 
                            c=clusters, cmap='viridis', alpha=0.6)
        plt.xlabel('log T90 (s)')
        plt.ylabel('log Fluence (erg/cm²)')
        plt.title('GRB Clusters (T90 & Fluence)', fontsize=16, fontweight='bold')
        plt.colorbar(scatter)
        
        # T90 distribution
        plt.subplot(2, 2, 2)
        for i in range(len(np.unique(clusters))):
            plt.hist(df['log_t90'][clusters == i], 
                    bins=30, alpha=0.5, 
                    label=f'Cluster {i}')
        plt.xlabel('log T90 (s)')
        plt.ylabel('Count')
        plt.title('T90 Distribution by Cluster', fontsize=16, fontweight='bold')
        plt.legend()
        
        # Fluence distribution
        plt.subplot(2, 2, 3)
        for i in range(len(np.unique(clusters))):
            plt.hist(df['log_fluence'][clusters == i], 
                    bins=30, alpha=0.5, 
                    label=f'Cluster {i}')
        plt.xlabel('log Fluence (erg/cm²)')
        plt.ylabel('Count')
        plt.title('Fluence Distribution by Cluster', fontsize=16, fontweight='bold')
        plt.legend()
        
        # Sky distribution
        plt.subplot(2, 2, 4)
        scatter = plt.scatter(df['ra'], df['dec'], 
                            c=clusters, cmap='viridis', alpha=0.6)
        plt.xlabel('RA (degrees)')
        plt.ylabel('Dec (degrees)')
        plt.title('Sky Distribution of Clusters', fontsize=16, fontweight='bold')
        plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.savefig(f'clusters_{feature_type}.png')
    plt.close()

def analyze_clusters(df, clusters, feature_type):
    output = f"\nCluster Analysis for {feature_type}:\n"
    output += "="*50 + "\n"
    
    for i in range(len(np.unique(clusters))):
        cluster_data = df[clusters == i]
        output += f"\nCluster {i}:\n"
        output += f"Number of GRBs: {len(cluster_data)}\n"
        output += f"T90 range: {np.exp(cluster_data['log_t90'].min()):.2e} to {np.exp(cluster_data['log_t90'].max()):.2e} s\n"
        output += f"Average T90: {np.exp(cluster_data['log_t90'].mean()):.2e} ± {np.exp(cluster_data['log_t90'].std()):.2e} s\n"
        output += f"Fluence range: {np.exp(cluster_data['log_fluence'].min()):.2e} to {np.exp(cluster_data['log_fluence'].max()):.2e} erg/cm²\n"
        output += f"Average Fluence: {np.exp(cluster_data['log_fluence'].mean()):.2e} ± {np.exp(cluster_data['log_fluence'].std()):.2e} erg/cm²\n"
    
    return output

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('GRB_Table.csv')
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Open file for writing results
    with open('output.txt', 'w') as f:
        # 1. Clustering based on T90
        X_t90 = StandardScaler().fit_transform(df[['log_t90']].values)
        sil_scores_t90 = plot_and_save_elbow(X_t90, 'T90')
        kmeans_t90, clusters_t90 = perform_kmeans(X_t90)
        plot_clustering_results(df, clusters_t90, "T90")
        f.write(analyze_clusters(df, clusters_t90, "T90"))
        
        # 2. Clustering based on Fluence
        X_fluence = StandardScaler().fit_transform(df[['log_fluence']].values)
        sil_scores_fluence = plot_and_save_elbow(X_fluence, 'Fluence')
        kmeans_fluence, clusters_fluence = perform_kmeans(X_fluence)
        plot_clustering_results(df, clusters_fluence, "Fluence")
        f.write(analyze_clusters(df, clusters_fluence, "Fluence"))
        
        # 3. Clustering based on both
        X_both = StandardScaler().fit_transform(df[['log_t90', 'log_fluence']].values)
        sil_scores_both = plot_and_save_elbow(X_both, 'Both')
        kmeans_both, clusters_both = perform_kmeans(X_both)
        plot_clustering_results(df, clusters_both, "Both")
        f.write(analyze_clusters(df, clusters_both, "Both"))
        
        # Save silhouette scores
        f.write("\nBest number of clusters based on Silhouette Score:\n")
        f.write(f"T90: {np.argmax(sil_scores_t90) + 2}\n")
        f.write(f"Fluence: {np.argmax(sil_scores_fluence) + 2}\n")
        f.write(f"Both: {np.argmax(sil_scores_both) + 2}\n")
        
        # Save cluster assignments to CSV
        df['cluster_t90'] = clusters_t90
        df['cluster_fluence'] = clusters_fluence
        df['cluster_both'] = clusters_both
        df.to_csv('GRB_Table_clustered.csv', index=False)

if __name__ == "__main__":
    main()