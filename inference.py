import os
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer
from src.models import CLIP
from src.dataset import get_preprocess_transforms, CLIP_COCO_Dataset

def load_trained_model(checkpoint_path, device):
    model = CLIP().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def zero_shot_inference(model , image , text_labels , preprocess , tokenizer , device):
    model.eval()

    image_input = preprocess(image.convert('RGB')).unsqueeze(0).to(device)
    text_inputs = tokenizer(
        text_labels,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        image_features = model.image_encoder(image_input)
        text_features = model.text_encoder(text_inputs['input_ids'], text_inputs['attention_mask'])

        image_proj = F.normalize(model.image_projection(image_features), p=2, dim=-1)
        text_proj = F.normalize(model.text_projection(text_features), p=2, dim=-1)

        # Compute dot product similarity
        logits = (image_proj @ text_proj.T).squeeze(0)
        probs = F.softmax(logits , dim=0)

    best_match_index = probs.argmax().item()
    
    # Visualization
    plt.figure(figsize=(8, 5))
    plt.imshow(image)
    plt.title(f"Prediction: {text_labels[best_match_index]}\nConfidence: {probs[best_match_index]:.4f}")
    plt.axis('off')
    plt.show()
    
    return text_labels[best_match_index], probs

def text_to_image_retrieval(model , query_text , image_embeddings , val_dataset , tokenizer , device , top_k=3):
    model.eval()

    text_inputs = tokenizer(
        [query_text],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        text_features = model.text_encoder(text_inputs['input_ids'] , text_inputs['attention_mask'])
        text_proj = F.normalize(model.text_projection(text_features) , p=2 , dim=-1)

        scores = (text_proj @ image_embeddings.T).squeeze(0)
        top_values, top_indices = scores.topk(top_k)

    print(f"Searching for: '{query_text}'")
    plt.figure(figsize=(15, 5))

    for i, idx in enumerate(top_indices):
        img_id = val_dataset.ids[idx.item()]
        img_metadata = val_dataset.coco.loadImgs(img_id)[0]
        img_path = os.path.join(val_dataset.root, img_metadata['file_name'])

        raw_img = Image.open(img_path).convert('RGB')

        plt.subplot(1, top_k, i + 1)
        plt.imshow(raw_img)
        plt.title(f"Rank {i+1}\nScore: {top_values[i].item():.4f}")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = "clip_best_model.pth"
    TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    PREPROCESS = get_preprocess_transforms()

    clip_model = load_trained_model(MODEL_PATH, DEVICE)