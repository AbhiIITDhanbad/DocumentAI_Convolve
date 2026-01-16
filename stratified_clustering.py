# smart_sampler.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
from pathlib import Path
from PIL import Image
import pytesseract
from collections import Counter

class IntelligentSampler:
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.images = list(self.image_dir.glob('*.png'))
        
    def extract_features(self, image_path):
        """
        Extract features that capture diversity:
        - Image quality (resolution, contrast)
        - Language (script detection)
        - Layout complexity (text density)
        - Visual characteristics (color, brightness)
        """
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # Feature 1: Resolution
        height, width = gray.shape
        features.extend([height, width, height * width])
        
        # Feature 2: Image quality metrics
        # Contrast (std of pixel values)
        contrast = np.std(gray)
        features.append(contrast)
        
        # Brightness (mean pixel value)
        brightness = np.mean(gray)
        features.append(brightness)
        
        # Feature 3: Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        features.append(sharpness)
        
        # Feature 4: Text density (what % of image is text)
        # Quick OCR to get text regions
        try:
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            text_areas = sum([w * h for w, h in zip(ocr_data['width'], ocr_data['height'])])
            text_density = text_areas / (height * width)
            features.append(text_density)
        except:
            features.append(0)
        
        # Feature 5: Color vs grayscale
        color_variance = np.std(img, axis=2).mean()
        features.append(color_variance)
        
        # Feature 6: Language hint (presence of non-ASCII chars)
        # We'll use a quick OCR sample
        try:
            text_sample = pytesseract.image_to_string(gray)[:500]
            # Check for Devanagari unicode range
            hindi_chars = sum(1 for c in text_sample if '\u0900' <= c <= '\u097F')
            gujarati_chars = sum(1 for c in text_sample if '\u0A80' <= c <= '\u0AFF')
            features.extend([hindi_chars, gujarati_chars])
        except:
            features.extend([0, 0])
        
        return np.array(features)
    
    def sample_diverse_subset(self, n_samples=100):
        """
        Use clustering to ensure diverse samples
        """
        print(f"Extracting features from {len(self.images)} images...")
        
        # Extract features for all images
        all_features = []
        valid_images = []
        
        for img_path in self.images:
            try:
                features = self.extract_features(img_path)
                all_features.append(features)
                valid_images.append(img_path)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
        
        features_array = np.array(all_features)
        
        # Normalize features (important for clustering)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_array)
        
        # Reduce dimensionality for visualization (optional)
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_normalized)
        
        # K-means clustering
        print(f"Clustering into {n_samples} groups...")
        kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
        kmeans.fit(features_normalized)
        
        # Select one sample from each cluster (closest to centroid)
        selected_indices = []
        for cluster_id in range(n_samples):
            # Get all points in this cluster
            cluster_mask = kmeans.labels_ == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Find point closest to centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(
                features_normalized[cluster_indices] - centroid, 
                axis=1
            )
            closest_idx = cluster_indices[distances.argmin()]
            selected_indices.append(closest_idx)
        
        selected_images = [valid_images[i] for i in selected_indices]
        
        # Save metadata for analysis
        metadata = {
            'features_2d': features_2d,
            'cluster_labels': kmeans.labels_,
            'selected_indices': selected_indices
        }
        
        return selected_images, metadata
    
    def visualize_sampling(self, metadata):
        """Visualize the sampling strategy"""
        import matplotlib.pyplot as plt
        
        features_2d = metadata['features_2d']
        labels = metadata['cluster_labels']
        selected = metadata['selected_indices']
        
        plt.figure(figsize=(12, 8))
        
        # Plot all points
        scatter = plt.scatter(
            features_2d[:, 0], 
            features_2d[:, 1],
            c=labels, 
            cmap='tab20',
            alpha=0.6,
            s=50
        )
        
        # Highlight selected points
        plt.scatter(
            features_2d[selected, 0],
            features_2d[selected, 1],
            c='red',
            marker='*',
            s=200,
            edgecolors='black',
            label='Selected for annotation'
        )
        
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Stratified Sampling: Selected Images Cover All Clusters')
        plt.legend()
        plt.colorbar(scatter, label='Cluster ID')
        plt.tight_layout()
        plt.savefig('sampling_visualization.png', dpi=300)
        plt.show()
        
        print(f"✓ Visualization saved to sampling_visualization.png")

# Usage
if __name__ == '__main__':
    sampler = IntelligentSampler(r'C:\Users\Abhi9\OneDrive\Documents\Convolve\train')
    
    # Sample 100 diverse images
    selected_images, metadata = sampler.sample_diverse_subset(n_samples=100)
    
    # Visualize
    sampler.visualize_sampling(metadata)
    
    # Save list for annotation
    with open('images_to_annotate.txt', 'w') as f:
        for img_path in selected_images:
            f.write(f"{img_path}\n")
    
    print(f"✓ Selected {len(selected_images)} images for manual annotation")
    print(f"✓ List saved to images_to_annotate.txt")





