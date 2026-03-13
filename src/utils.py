import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from PIL import Image

def get_batch_embeddings(model , batch , device):
    model.eval()
    with torch.no_grad():
        image_features = model.image_encoder(batch['pixel_values'].to(device))
        text_features = model.text_encoder(batch['input_ids'].to(device) , batch['attention_mask'].to(device))

        image_proj = F.normalize(model.image_projection(image_features) , p=2 , dim=-1)
        text_proj = F.normalize(model.text_projection(text_features) , p=2 , dim=-1)

    return image_proj , text_proj

def plot_tsne(image_embeddings , text_embeddings , epoch , save_path , num_samples=25):
    num_samples = min(num_samples , image_embeddings.shape[0])

    image_subset = image_embeddings[:num_samples]
    text_subset = text_embeddings[:num_samples]

    embeddings = np.concatenate([image_subset , text_subset] , dim=0).cpu().numpy()

    tsne = TSNE(n_components=2 , perplexity=min(15,num_samples-1) , init='pca' , random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    image_reduced = reduced_embeddings[:num_samples]
    text_reduced = reduced_embeddings[num_samples:]

    # Visualization
    plt.figure(figsize=(10,8))
    plt.scatter(image_reduced[:,0] , image_reduced[:,1] , alpha=0.7 , label='Images' , c='dodgerblue' , s=100)
    plt.scatter(text_reduced[:,0] , text_reduced[:,1] , alpha=0.7 , label='Texts' , c='crimson' , marker='X' , s=100)

    for i in range(num_samples):
        plt.plot(
            [image_reduced[i,0] , text_reduced[i,0]],
            [image_reduced[i,1] , text_reduced[i,1]],
            color='gray',
            linestyle='--',
            alpha=0.3,
        )

    plt.title(f"t-SNE Projection: Multimodal Alignment (Epoch {epoch})" , fontsize=15 , fontweight='bold')
    plt.legend()
    plt.grid(True , alpha=0.1)
    
    plt.savefig(save_path , bbox_inches='tight')
    plt.close()

def similarity_matrix(logits , epoch , save_path):
    plt.figure(figsize=(10,8))

    # The perfect similarity matrix is the identity matrix
    sns.heatmap(logits.cpu().numpy() , cmap='viridis')
    plt.title(f'Image-Text Similarity Matrix (Epoch {epoch})' , fontsize=15 , fontweight='bold')
    plt.xlabel('Text Embeddings')
    plt.ylabel('Image Embeddings')

    plt.savefig(save_path , bbox_inches='tight')
    plt.close()

def create_gif(input_folder , output_filename , duration=800):
    ...

def calculate_metrics(logits):
    labels = torch.arange(logits.shape[0]).to(logits.device)

    top1_acc = (logits.argmax(dim=1) == labels).float().mean().item()

    _ , top5_indices = torch.topk(5 , dim=1)
    top5_acc = (top5_indices == labels[:,None]).float().mean().item()

    return top1_acc , top5_acc