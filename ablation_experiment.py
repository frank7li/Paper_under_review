
import pandas as pd
import random
import cv2
import numpy as np
import torch
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from boxmot.trackers.hybridsort.hybridsort import HybridSORT
from collections import defaultdict
import json
import shutil
import time
import math
import torchreid
import torch.nn as nn
import argparse

# Load gallery features
def load_gallery_features(appearance_model, num_people, data_dict):
    feature_dict = {}
    i = 1
    for image_name in data_dict:
        if i % 1000 == 0:
            print(f"loaded {i}/{len(data_dict)}")
        i += 1
        p_id = data_dict[image_name]['id']
        img_path = os.path.join(folder_path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            if image_name != ".DS_Store":
                print(f"Warning: Image at {img_path} could not be loaded.")
            continue
        img = cv2.resize(img, (128, 256))
        img = torch.tensor(img).float().div(255).unsqueeze(0).permute(0, 3, 1, 2)
        img = img.to(device)
        with torch.no_grad():
            feature = appearance_model(img).cpu().numpy()

        feature_dict[image_name] = feature
    return feature_dict


# Function to extract features for a given image
def extract_features(model, img):
    img = cv2.resize(img, (128, 256))
    img = torch.tensor(img).float().div(255).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        features = model(img).cpu().numpy()
    return features


# Load gallery features
def load_gallery_features_mlfn(appearance_model, num_people, data_dict):
    feature_dict = {}
    i = 1
    for image_name in data_dict:
        if i % 1000 == 0:
            print(f"loaded {i}/{len(data_dict)}")
        i += 1
        p_id = data_dict[image_name]['id']
        img_path = os.path.join(folder_path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            if image_name != ".DS_Store":
                print(f"Warning: Image at {img_path} could not be loaded.")
            continue
        feature = extract_features_mlfn(appearance_model, img)
        feature_dict[image_name] = feature
    return feature_dict

def extract_features_mlfn(model, img):
    """Extracts a feature vector from an image using MLFN."""
    # Resize and convert to PyTorch tensor
    img = cv2.resize(img, (128, 256))
    img = torch.tensor(img, dtype=torch.float32).div(255)  # Normalize to [0,1]
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)  # Convert to (batch, channels, height, width)
    with torch.no_grad():
        feature_vector = model(img)  # Extract feature
    return feature_vector.cpu().numpy().reshape(1, -1)  # Convert to 2D array



# Load gallery features
def load_gallery_features_resnet(appearance_model, num_people, data_dict):
    feature_dict = {}
    i = 1
    for image_name in data_dict:
        if i % 1000 == 0:
            print(f"loaded {i}/{len(data_dict)}")
        i += 1
        p_id = data_dict[image_name]['id']
        img_path = os.path.join(folder_path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            if image_name != ".DS_Store":
                print(f"Warning: Image at {img_path} could not be loaded.")
            continue
        feature = extract_features_resnet(appearance_model, img)
        feature_dict[image_name] = feature
    return feature_dict


def extract_features_resnet(model, img):
    """Extracts a 512-dim feature vector from an image."""
    # Resize and convert to PyTorch tensor
    img = cv2.resize(img, (128, 256))
    img = torch.tensor(img, dtype=torch.float32).div(255)  # Ensure dtype is float32
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)  # Move to (batch, channels, height, width)
    with torch.no_grad():
        feature_vector = model(img)  # Extract feature
    return feature_vector.cpu().numpy().reshape(1, -1)  # Convert (512,) → (1, 512)


def compute_metrics(appearance_model, feature_dict, num_people, result_path, error_path, setting, model_name, test_dict, train_dict, fold_num):
    rank_1_accuracy = []
    rank_5_accuracy = []
    average_precision = []
    correct_log = {}
    incorrect_log = {}
    # Extract feature matrices for batch similarity computation
    test_images = list(test_dict.keys())
    train_images = list(train_dict.keys())

    # Extract features while handling potential missing feature cases
    test_features = np.array([feature_dict[img].squeeze() for img in test_images if img in feature_dict])
    gallery_features = np.array([feature_dict[img].squeeze() for img in train_images if img in feature_dict])

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(test_features, gallery_features)

    # Get sorted indices (descending order) for ranking
    ranked_indices = np.argsort(-similarity_matrix, axis=1)

    for query_idx, image_name in enumerate(test_images):
        person_id = test_dict[image_name]['id']
        # Get ranked gallery image indices
        sorted_gallery_indices = ranked_indices[query_idx]
        # Retrieve sorted similarity scores, gallery image IDs, and image names
        sorted_similarities = similarity_matrix[query_idx][sorted_gallery_indices]
        sorted_gallery_names = [train_images[idx] for idx in sorted_gallery_indices]
        sorted_gallery_ids = [train_dict[img]['id'] for img in sorted_gallery_names]
        # Rank-1 Accuracy
        rank_1 = sorted_gallery_ids[0] == person_id
        # Rank-5 Accuracy
        rank_5 = person_id in sorted_gallery_ids[:5]
        # Compute Average Precision (AP)
        tp = 0
        precisions = []
        for i, gal_img_id in enumerate(sorted_gallery_ids):
            if gal_img_id == person_id:
                tp += 1
                precision = tp / (i + 1)
                precisions.append(precision)

        ap = np.mean(precisions) if precisions else 0
        top_match = sorted_gallery_names[0]

        # Log results
        if rank_1:
            correct_log[image_name] = {
                "predicted": sorted_gallery_ids[0],
                "highest_similarity_image": top_match,
                "similarity": float(sorted_similarities[0]),
                "AP": ap
            }
        else:
            incorrect_log[image_name] = {
                "predicted": sorted_gallery_ids[0],
                "highest_similarity_image": top_match,
                "similarity": float(sorted_similarities[0]),
                "AP": ap,
                "rank_5": rank_5
            }

        rank_1_accuracy.append(rank_1)
        rank_5_accuracy.append(rank_5)
        average_precision.append(ap)

    # Compute final evaluation metrics
    final_metrics = {
        'rank_1_accuracy': np.mean(rank_1_accuracy) if rank_1_accuracy else 0.0,
        'rank_5_accuracy': np.mean(rank_5_accuracy) if rank_5_accuracy else 0.0,
        'mAP': np.mean(average_precision) if average_precision else 0.0,
    }

    result = {
        "model_name": model_name,
        "gallery_size": len(train_dict),
        "setting": setting,
        "fold": fold_num,
        "metric_results": final_metrics,
        "correct": correct_log,
        "incorrect": incorrect_log
    }

    try:
        with open(result_path, "r") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = []

    all_results.append(result)

    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=4)

    return final_metrics




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Model")
    parser.add_argument('--fold', type=str, help="Fold section'")
    args = parser.parse_args()
    model_name = args.model
    fold_section = args.fold

    folder_path = "images"
    file_path = 'labeling.xlsx'  

    if fold_section == "1":
        fold_numbers = [0]
    if fold_section == "2":
        fold_numbers = [1,2,3]
    elif fold_section == "3":
        fold_numbers = [4,5,6]
    elif fold_section == "4":
        fold_numbers = [7,8,9]

    print(f"{model_name}_{fold_section}")

    result_path = f"{file_path[:-5]}_results_{model_name}_fold{fold_section}.json"

    device = "cpu"

    # Enable cuDNN optimizations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.deterministic = False

    # Load the Excel file
    excel_data = pd.read_excel(file_path)

    # Calculate median for resolution and occlusion for abaltion study threshold
    resolution_median = excel_data['image_area'].median()
    occlusion_median = excel_data['occlusion_ratio'].median()
    image_dict = excel_data.set_index('img_names').T.to_dict()


    num_people = 16

    create_file = []
    with open(result_path, "w") as f:
        json.dump(create_file, f, indent=4)

    # Set pretrained=True and remove the load_pretrained_weights to use pretrained models for MSMT
    if model_name == "osnet_ain":
        appearance_model = torchreid.models.build_model(
        name='osnet_ain_x1_0', 
        num_classes=0, 
        loss='softmax',
        pretrained=False, 
        use_gpu=False
        )
        trained_weights_path = Path("osnet_ain_x1_0_cross_entropy_5epoch_angle2-cpu.pt")
        torchreid.utils.load_pretrained_weights(appearance_model, trained_weights_path) 
        appearance_model = appearance_model.to(device)
        appearance_model.eval()
    elif model_name == "osnet":
        appearance_model = torchreid.models.build_model(
        name='osnet_x1_0', 
        num_classes=0, 
        loss='softmax',
        pretrained=False, 
        use_gpu=False
        )
        trained_weights_path = "osnet-finetuned-weights-cpu.pt" 
        torchreid.utils.load_pretrained_weights(appearance_model, trained_weights_path)
        appearance_model = appearance_model.to(device)
        appearance_model.eval()
    elif model_name == "mlfn":
        appearance_model = torchreid.models.build_model(
        name='mlfn', 
        num_classes=0, 
        loss='softmax',
        pretrained=False, 
        use_gpu=False
        )
        trained_weights_path = "mlfn-finetuned_weights-cpu.pt" 
        torchreid.utils.load_pretrained_weights(appearance_model, trained_weights_path)
        appearance_model = appearance_model.to(device)
        appearance_model.eval()
    elif model_name == "resnet50":
        appearance_model = torchreid.models.build_model(
        name='resnet50_fc512', 
        num_classes=0, 
        loss='softmax',
        pretrained=False, 
        use_gpu=False
        )
        trained_weights_path = "resnet50-finetuned-weights-cpu.pt"
        torchreid.utils.load_pretrained_weights(appearance_model, trained_weights_path)
        appearance_model = appearance_model.to(device)
        appearance_model.eval()


    elif model_name == "mobilenet":

        appearance_model = torchreid.models.build_model(
        name='mobilenetv2_x1_4',  
        num_classes=0,  
        loss='softmax',
        pretrained=False, 
        use_gpu=False
        )

        trained_weights_path = "/home/frankli/person-reidentification/mobilenet-trained-weights-cpu.pt" 
        torchreid.utils.load_pretrained_weights(appearance_model, trained_weights_path)
        # Move model to GPU and set to evaluation mode
        appearance_model = appearance_model.to(device)
        appearance_model.eval()

    if model_name == "mlfn":
        all_features = load_gallery_features_mlfn(appearance_model, num_people, image_dict)
    elif model_name == "resnet50":
        all_features = load_gallery_features_resnet(appearance_model, num_people, image_dict)
    else:
        all_features = load_gallery_features(appearance_model, num_people, image_dict)
        
    print("finished loading gallery")

    settings = ['baseline', 'spread', 'lighting', 'front', 'left', 'right', 'back', 'occlusion_high', 'occlusion_low', 'resolution_high', 'resolution_low']


    seed_number = 1000
    random.seed(seed_number)
    keys = list(image_dict.keys())
    random.shuffle(keys)
    keys_splits = np.array_split(keys, 10)
    for fold in fold_numbers:
        train = []
        test = []
        
        train = [key for i in range(10) if i != fold for key in keys_splits[i]] 
        test = list(keys_splits[fold])

        t_dict = {key: image_dict[key] for key in train}
        test_dict = {key: image_dict[key] for key in test}


        for setting in settings:
            train_dict = {}
            if setting =='baseline':
                train_dict = t_dict
            elif setting == 'front':
                train_dict = {key: value for key, value in t_dict.items() if value.get('orientation') != 'Front'}
            elif setting == 'left':
                train_dict = {key: value for key, value in t_dict.items() if value.get('orientation') != 'Left'}
            elif setting == 'right':
                train_dict = {key: value for key, value in t_dict.items() if value.get('orientation') != 'Right'}
            elif setting == 'back':
                train_dict = {key: value for key, value in t_dict.items() if value.get('orientation') != 'Back'}
            elif setting == 'lighting':
                train_dict = {key: value for key, value in t_dict.items() if value.get('lighting_condition')!= 'L'}
            elif setting == 'spread':
                train_dict = {key: value for key, value in t_dict.items() if value.get('pose_variation') != 'Spread'}
            elif setting == 'occlusion_high': 
                train_dict = {key: value for key, value in t_dict.items() if value.get('occlusion_ratio') > occlusion_median}
            elif setting == 'occlusion_low': 
                train_dict = {key: value for key, value in t_dict.items() if value.get('occlusion_ratio') <= occlusion_median}
            elif setting == 'resolution_high':
                train_dict = {key: value for key, value in t_dict.items() if value.get('image_area') > resolution_median}
            elif setting == 'resolution_low':
                train_dict = {key: value for key, value in t_dict.items() if value.get('image_area') <= resolution_median} 

            # Compute metrics
            metrics = compute_metrics(appearance_model, all_features, num_people, result_path, error_path, setting, model_name, test_dict, train_dict, fold)
        
            print(f"model: {model_name}")
            print(f"fold: {fold}")
            print(f"setting: {setting}")
            print(f"Rank-1 Accuracy: {metrics['rank_1_accuracy']:.4f}")
            print(f"Rank-5 Accuracy: {metrics['rank_5_accuracy']:.4f}")
            print(f"mAP: {metrics['mAP']:.4f}")
            print("______________________________")
            print()


