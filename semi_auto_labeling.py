import cv2
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
import math

# Define the device to use (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pose_model = YOLO('yolov8n-pose.pt')

def project_point_onto_vector(point, vector):
    p = np.array(point) 
    v = np.array(vector)
    projection_scalar = np.dot(p, v) / np.dot(v, v)
    intersection_point = projection_scalar * v
    
    return intersection_point

def classify_orientation(keypoints):
    if len(keypoints) < 6:
        print("Insufficient keypoints detected to classify orientation.")
        return ("Uncertain", "Uncertain")

    left_shoulder = keypoints[5][0]
    right_shoulder = keypoints[6][0]
    nose = keypoints[0][0]

    # Calculate shoulder and nose relationships
    shoulder_diff = left_shoulder - right_shoulder
    nose_to_left_shoulder = nose - left_shoulder
    nose_to_right_shoulder = nose - right_shoulder

    shoulder_diff_vector = [shoulder_diff, keypoints[5][1] - keypoints[6][1]]
    nose_point = [keypoints[0][0], keypoints[0][1]]
    intersection_point = project_point_onto_vector(nose_point, shoulder_diff_vector)
    projection_dist_to_left = math.sqrt((left_shoulder - intersection_point[0]) * (left_shoulder - intersection_point[0]) + (keypoints[5][1] - intersection_point[1])*(keypoints[5][1] - intersection_point[1]))
    projection_dist_to_right = math.sqrt((right_shoulder - intersection_point[0]) * (right_shoulder - intersection_point[0]) + (keypoints[6][1] - intersection_point[1])*(keypoints[6][1] - intersection_point[1]))

    spread = "None"
    if len(keypoints) > 9:

        face_only_img = False

        dist_eye_to_shoulder = abs(keypoints[1][1]- keypoints[5][1])
        dist_shoulder_to_ankle = abs(keypoints[5][1]- keypoints[15][1])
        
        if dist_shoulder_to_ankle < 3 * dist_eye_to_shoulder:
            face_only_img = True

        dist_threshold_l = abs(keypoints[5][1]-keypoints[9][1]) * 0.8
        dist_threshold_r = abs(keypoints[6][1]-keypoints[10][1]) * 0.8
        left_wrist = keypoints[9][0]
        right_wrist = keypoints[10][0]
        if abs(left_wrist - left_shoulder) > dist_threshold_l or abs(right_wrist - right_shoulder) > dist_threshold_r:
            spread = "Spread"

        if face_only_img:
            spread = "None"

    if nose_to_left_shoulder > 0 and nose_to_right_shoulder > 0:
        return ("Right", spread)
    elif nose_to_left_shoulder < 0 and nose_to_right_shoulder < 0:
        return ("Left",spread)
    elif projection_dist_to_left >  1.6 * projection_dist_to_right:
        return("Left", spread)
    elif projection_dist_to_right >  1.6 * projection_dist_to_left:
        return("Right", spread)
    elif shoulder_diff >0 and nose_to_left_shoulder < 0  and nose_to_right_shoulder > 0:
        return ("Front", spread)
    elif shoulder_diff <0 and nose_to_left_shoulder > 0  and nose_to_right_shoulder < 0:
        return ("Back", spread)
    else:
        return ("Uncertain", spread)



def annotate_lighting_condition(id_folder_path, id_folder):
    """
    Annotates each image in the folder with its lighting condition ('low', 'medium', or 'high')
    and a lighting ratio based on the combined score of brightness, contrast, and edge density.
    Also displays a boxplot for the combined lighting scores.
    """
    lighting_data = []
    img_paths = []

    for img_name in sorted(os.listdir(id_folder_path)):
        img_path = os.path.join(id_folder_path, img_name)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # Convert to grayscale and calculate lighting metrics
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray_img)
                contrast = np.std(gray_img)
                edges = cv2.Canny(gray_img, 100, 200)
                edge_density = np.sum(edges > 0) / (gray_img.shape[0] * gray_img.shape[1])
                lighting_data.append((img_path, mean_brightness, contrast, edge_density))
                img_paths.append(img_path)

    if lighting_data:
        brightness_data = [ld[1] for ld in lighting_data]
        contrast_data = [ld[2] for ld in lighting_data]
        edge_density_data = [ld[3] for ld in lighting_data]

        df_lighting = pd.DataFrame({
            'brightness': brightness_data,
            'contrast': contrast_data,
            'edge_density': edge_density_data
        })
        combined_score = df_lighting['brightness'] + df_lighting['contrast'] + df_lighting['edge_density']
        q1_combined = combined_score.quantile(0.25)
        q3_combined = combined_score.quantile(0.75)

        max_combined_score = combined_score.max()
        lighting_ratios = combined_score / max_combined_score if max_combined_score > 0 else 0

        lighting_annotations = []
        for (img_path, mean_brightness, contrast, edge_density), combined_value, lighting_ratio in zip(lighting_data, combined_score, lighting_ratios):
            if combined_value <= q1_combined:
                lighting_condition = 'low'
            elif combined_value > q3_combined:
                lighting_condition = 'high'
            else:
                lighting_condition = 'medium'
            lighting_annotations.append((img_path, lighting_condition, lighting_ratio))
        return lighting_annotations
    else:
        return []