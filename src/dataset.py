import random
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader

def get_preprocess_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485 , 0.456 , 0.406],
            std=[0.229 , 0.224 , 0.225]
        )
    ])

def get_dataloader(root_dir , tokenizer , batch_size=128):
    preprocess = get_preprocess_transforms()

    # Create the datasets
    train_dataset = CLIP_COCO_Dataset(
        root=f'{root_dir}/images/train2017',
        annFile=f'{root_dir}/annotations/train2017.json',
        preprocess=preprocess,
        tokenizer=tokenizer
    )
    val_dataset = CLIP_COCO_Dataset(
        root=f'{root_dir}/images/val2017',
        annFile=f'{root_dir}/annotations/val2017.json',
        preprocess=preprocess,
        tokenizer=tokenizer
    )

    # Create the dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    return train_loader , val_loader , val_dataset


class CLIP_COCO_Dataset(dset.CocoCaptions):
    def __init__(self , root , annFile , preprocess , tokenizer):
        super().__init__(root , annFile)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __getitem__(self , index):
        image , captions = super().__getitem__(index)
        caption = random.choice(captions)  # Choosing a random caption
        image_input = self.preprocess(image.convert('RGB'))

        # Tokenize the caption
        text_input = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=77,
            return_tensors='pt'
        )

        return {
            'pixel_values' : image_input,
            'input_ids': text_input['input_ids'].squeeze(0),
            'attention_mask': text_input['attention_mask'].squeeze(0)
        }