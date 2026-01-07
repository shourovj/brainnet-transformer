import torch
import scipy.io as sio
import sentencepiece
import tiktoken
import einops
import wandb
import accelerate
import pandas as pd
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm





def get_file_list_prepare_csv(data_dir):
    all_files = glob(data_dir +'*.mat')
    print((all_files[0:5]))

    label_file = [file_name if file_name.split('.')[0].split('/')[-1] == 'label' else None for file_name in all_files ]
    label_file = [file_name for file_name in label_file if file_name is not None]

    # Extract the single label file (should be only one)
    if len(label_file) == 0:
        raise FileNotFoundError("No label file found in the data directory")
    label_file = label_file[0]  # Get the first (and should be only) label file


    feature_file = [file_name if file_name.split('.')[0].split('_')[-1] == 'feature' else None for file_name in all_files ]
    feature_file = [file_name for file_name in feature_file if file_name is not None]
    sorted_feature_file = sorted(feature_file, key=lambda x: int(x.split('.')[0].split('_')[2]))
    # print(sorted_feature_file)

    # len(feature_file)

    cluster_file = [file_name if file_name.split('.')[0].split('_')[-1] == 'index' else None for file_name in all_files ]
    cluster_file = [file_name for file_name in cluster_file if file_name is not None]
    sorted_cluster_file = sorted(cluster_file, key=lambda x: int(x.split('.')[0].split('_')[2]))
    # print(sorted_cluster_file)


    label_data = sio.loadmat(label_file)
    print(label_data.keys())
    labels = label_data['label']
    # print(labels)


    # Create DataFrame using dictionary - each key becomes a column
    # sorted_feature_file and sorted_cluster_file are lists of file paths (500 elements each)
    # labels is a numpy array of shape (500, 1), so we flatten it to (500,)
    df = pd.DataFrame({
        'feature_file': sorted_feature_file,
        'cluster_file': sorted_cluster_file,
        'label': labels.flatten()  # Flatten from (500, 1) to (500,)
    })
    # len(df)
    # df.to_csv('data.csv', index=False)





    df = pd.read_csv('./data_split/data.csv')
    print(f"Original dataset: {len(df)} samples")


    shape_dict = {}
    problematic_indices = []
    expected_shape = (400, 1632)

    for idx in tqdm(range(len(df))):
        row = df.iloc[idx]
        feature_file = data_dir + row['feature']
        
        try:
            f_data = sio.loadmat(feature_file)
            f_mat = f_data['feature_mat']
            shape = f_mat.shape
            
            if shape not in shape_dict:
                shape_dict[shape] = []
            shape_dict[shape].append(idx)
            
            # Check if shape doesn't match expected
            if shape[0] != expected_shape[0]:
                problematic_indices.append(idx)
        except Exception as e:
            print(f"Error loading {feature_file}: {e}")
            problematic_indices.append(idx)

    print(f"\nFound {len(shape_dict)} different shapes:")
    for shape, indices in sorted(shape_dict.items()):
        print(f"  Shape {shape}: {len(indices)} files")
        if shape[0] != expected_shape[0]:
            print(f"NON-STANDARD! First 5 files:")
            for i in indices[:5]:
                print(f"      - Row {i}: {df.iloc[i]['feature']}")

    # print(f"FILTERING OUT {len(problematic_indices)} PROBLEMATIC FILES...")

    df_clean = df.drop(index=problematic_indices).reset_index(drop=True)
    print(df.head())
    print(df_clean.head())
    df_clean.to_csv('data_split/data_clean.csv', index=False)
    print(f"Cleaned dataset: {len(df_clean)} samples (removed {len(problematic_indices)} files)")
    print(f"Original: {len(df)} → Cleaned: {len(df_clean)}")




    RANDOM_SEED = 42

    train_val_df, test_df = train_test_split(
        df_clean, 
        test_size=0.15, 
        random_state=RANDOM_SEED,
        stratify=df_clean['label'] 
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=15/85,  
        random_state=RANDOM_SEED,
        stratify=train_val_df['label']  
    )


    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df_clean)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df_clean)*100:.1f}%)")
    print(f"\nTrain label distribution:\n{train_df['label'].value_counts().sort_index()}")
    print(f"\nVal label distribution:\n{val_df['label'].value_counts().sort_index()}")
    print(f"\nTest label distribution:\n{test_df['label'].value_counts().sort_index()}")

    # Save the problematic files list for reference
    if problematic_indices:
        problematic_df = df.iloc[problematic_indices]
        problematic_df.to_csv('toy_data/split/problematic_files.csv', index=False)
        print(f"\n💾 Problematic files saved to 'toy_data/split/problematic_files.csv'")


    # Save cleaned splits
    train_df.to_csv('data_split/train_df.csv', index=False)
    val_df.to_csv('data_split/val_df.csv', index=False)
    test_df.to_csv('data_split/test_df.csv', index=False)


if __name__ == "__main__":
    data_dir = "toy_data/"

    get_file_list_prepare_csv(data_dir)
