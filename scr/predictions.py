# -*- coding: utf-8 -*-

"""

   This script is primarily used for evaluating. The duties of the prediction.py script are as follows:

     1. Argument Parsing: It begins by parsing command line arguments. 
        This allows users to specify various parameters like data paths, dataset names, number of workers, model save paths, etc.

     2. Data Loading and Preparation: The script loads a test dataset using a custom dataset class MammoEvaluation 
        and prepares a DataLoader for efficient batch processing. This is crucial for handling large datasets 
        and feeding them to the model in manageable chunks.

     3. Model Loading and Preparation: It loads a pre-trained model (and an older version of the model for comparison) 
        from specified paths. The models are wrapped in DataParallel for multi-GPU support, enhancing performance.

     4. Metric Initialization: Various evaluation metrics such as IoU (Intersection over Union), Precision, Recall, Accuracy, 
        and F-score are initialized. These metrics are essential for quantitatively assessing the model's performance.

     5. Evaluation Loop: The script enters a loop where it processes each batch of data from the test dataset. For each batch, it:
            - Moves the images and masks to the GPU for faster computation.
            - Generates predictions for breast and dense tissue using the current and old model.
            - Calculates various metrics by comparing the predictions to the ground truth masks.
            - Computes the area of breast and dense tissues in the predictions to determine tissue density.
            - Logs the metric values and density calculations for each image.

     6. Statistical Analysis: After processing all batches, the script calculates 
        and logs the mean and confidence interval for each metric and the density measurements. 
        This provides a statistical summary of the model's performance across the entire test dataset.

     7. Logging: Throughout the process, results are logged into two files - one for general metrics 
        and another for density differences. The script ensures that the files are properly initialized with headers 
        and that existing content is not overwritten.


"""


# Importing necessary libraries
import argparse  # Used for parsing command line arguments
import torch  # PyTorch library for deep learning
import torch.nn as nn  # Neural network module from PyTorch
from dataset import MammoDataset, MammoEvaluation, construct_val_set  # Custom dataset class for mammogram evaluation
from torch.utils.data import DataLoader, Subset
from utils import Config  # DataLoader class for batch processing
import segmentation_models_multi_tasking as smp  # Library for segmentation models
from tqdm import tqdm  # Library for progress bars
import numpy as np  # Numerical library for array operations
import scipy.stats as stats  # Library for statistical functions
import warnings  # Module to handle warnings

warnings.filterwarnings("ignore")  # Ignoring warnings
import requests  # Module to handle HTTP requests
from pathlib import Path  # Library for handling filesystem paths
import os  # Module for interacting with the operating system
import csv


def process_submission_mode(dataloader, model):
    output_path = "test_output/submission/submission.csv"
    with open(output_path, "w") as file:
        file.write("Filename,Density\n")

        for img_id, image in tqdm(dataloader):
            image = image.cuda()
            # Obtain predictions for both breast and dense tissue
            pred_b_mask, pred_d_mask = model.module.predict(image)

            # Convert predictions to CPU and numpy format for processing
            pred_b_mask = pred_b_mask[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]
            pred_d_mask = pred_d_mask[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]

            # Apply the threshold to the predictions
            threshold = 0.5

            # Convert predictions to binary masks using the same threshold
            pred_b_mask_binary = (pred_b_mask > threshold).astype(np.float32)
            pred_d_mask_binary = (pred_d_mask > threshold).astype(np.float32)

            # Calculate areas for breast and dense tissues from predictions
            breast_area = np.sum(np.array(pred_b_mask_binary) == 1)
            dense_area = np.sum(np.array(pred_d_mask_binary) == 1)

            # Calculate the density percentage
            density = round(((dense_area / breast_area) * 100), 3) if breast_area > 0 else 0

            # Write the result to the output file
            file.write(f"{img_id[0]},{density}\n")


def load_ground_truths(csv_path):
    ground_truths = {}
    with open(csv_path, mode="r") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # Skip the header
        ground_truths = {rows[0]: float(rows[1]) for rows in reader}
    return ground_truths


def process_testsubmission_mode(dataloader, model, ground_truths):

    IOU = smp.utils.metrics.IoU()
    Precision = smp.utils.metrics.Precision()
    Recall = smp.utils.metrics.Recall()
    Accuracy = smp.utils.metrics.Accuracy()
    Fscore = smp.utils.metrics.Fscore()

    absolute_errors = []

    test_header_indexs = {"B_Prec": [], "B_Rec": [], "B_F-Sc": [], "B_Acc": [], "B_IoU": [], "D_Prec": [], "D_Rec": [], "D_F-Sc": [], "D_Acc": [], "D_IoU": []}

    # Lists to keep track of the density values computed from the predicted masks
    new_dense_values = []

    # Extract the metric names from the dictionary keys to use in the log files' headers
    test_header_names = test_header_indexs.keys()

    # Open a file for appending results; one for general metrics
    with open("test_output/test/report.txt", "a+") as logs_file, open("test_output/test/report_density_difference.txt", "a+") as density_file:
        # Reading existing content to avoid overwriting
        logs_file.seek(0)
        existing_content = logs_file.read().strip()

        density_file.seek(0)
        dense_existing_content = density_file.read().strip()

        # If the file is empty or does not start with the expected headers, write the headers
        if not existing_content.startswith("Abbreviations"):
            # Headers for the logs file, including abbreviations for readability
            abbreviations = "Abbreviations\nB = Breast\nD = Dense\nPrec = Precision \nRec = Recall\nF-Sc = Fscore\nAcc = Accuracy\n-------------------"
            header = "Image_ID\t\t\t\t" + "\t".join(test_header_names)
            logs_file.write(abbreviations + "\n")
            logs_file.write(header + "\n")

        if not dense_existing_content.startswith("Image_ID"):
            dense_header = "Image_ID\t\t\t\tPredicted Density\t\tGround Truth\t\tDifference"
            density_file.write(dense_header + "\n")

        # Loop over each batch of data provided by the dataloader
        for img_id, image, b_mask_org, d_mask_org in tqdm(dataloader):

            # Move the images and masks to the GPU for computation
            image = image.cuda()
            b_mask_org = b_mask_org.cuda()
            d_mask_org = d_mask_org.cuda()

            # Generate predictions for the images
            pred_b_mask, pred_d_mask = model.module.predict(image)

            # Calculate the metrics and store them
            for metric in test_header_names:
                if metric.startswith("B_P"):
                    value = Precision(pred_b_mask, b_mask_org)
                elif metric.startswith("B_R"):
                    value = Recall(pred_b_mask, b_mask_org)
                elif metric.startswith("B_F"):
                    value = Fscore(pred_b_mask, b_mask_org)
                elif metric.startswith("B_A"):
                    value = Accuracy(pred_b_mask, b_mask_org)
                elif metric.startswith("B_I"):
                    value = IOU(pred_b_mask, b_mask_org)
                elif metric.startswith("D_P"):
                    value = Precision(pred_d_mask, d_mask_org)
                elif metric.startswith("D_R"):
                    value = Recall(pred_d_mask, d_mask_org)
                elif metric.startswith("D_F"):
                    value = Fscore(pred_d_mask, d_mask_org)
                elif metric.startswith("D_A"):
                    value = Accuracy(pred_d_mask, d_mask_org)
                else:
                    value = IOU(pred_d_mask, d_mask_org)

                value = round(value.item(), 3)
                test_header_indexs[metric].append(value)

            # Calculate the density and store the value
            pred_b_mask = pred_b_mask[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]
            pred_d_mask = pred_d_mask[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]

            # Apply the threshold to the predictions
            threshold = 0.5

            # Convert predictions to binary masks using the same threshold
            pred_b_mask_binary = (pred_b_mask > threshold).astype(np.float32)
            pred_d_mask_binary = (pred_d_mask > threshold).astype(np.float32)

            breast_area = np.sum(pred_b_mask_binary == 1)
            dense_area = np.sum(pred_d_mask_binary == 1)

            new_density = round((dense_area / breast_area) * 100, 3) if breast_area > 0 else 0
            new_dense_values.append(new_density)

            # Log the results
            metric_values = [test_header_indexs[metric][-1] for metric in test_header_names]
            row = f"{img_id[0]}\t" + "\t".join(map(str, metric_values))
            logs_file.write(row + "\n")

            # Look up the ground truth density
            ground_truth_density = ground_truths.get(img_id[0], 0)
            difference = new_density - ground_truth_density
            absolute_error = abs(new_density - ground_truth_density)
            absolute_errors.append(absolute_error)
            dense_row = f"{img_id[0]}\t{new_density}\t\t\t{ground_truth_density}\t\t\t{difference:.3f}"
            density_file.write(dense_row + "\n")

        mean_absolute_error = np.mean(absolute_errors)
        density_file.write(f"Mean Absolute Error: {mean_absolute_error:.3f}\n")

        # Calculate and log the mean and confidence interval for the density values
        def mean_cal(data):
            return round(np.mean(data), 3)

        def average_count(data):
            mean = mean_cal(data)
            ci_min, ci_max = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=stats.sem(data))
            return round((ci_max - ci_min) / 2, 3)

        avg_density = mean_cal(new_dense_values)
        ci_density = average_count(new_dense_values)
        logs_file.write(f"\nAverage Density: {avg_density}\n")
        logs_file.write(f"Density Confidence Interval: {ci_density}\n")


def main():
    config = Config()
    PREDICTION_MODEL_PATH = os.path.join(config.PROJECT_ROOT, "test_output/models/", f"{config.model_name}.pth")

    ground_truths = load_ground_truths(os.path.join(config.PROJECT_ROOT, config.train_data_path, "../../train.csv"))
    ground_truths_path = os.path.join(config.PROJECT_ROOT, config.train_data_path, "../../train.csv")

    # Depending on the mode, the dataset directory paths will be set appropriately in the MammoEvaluation class.
    if config.prediction_mode == "submission":
        test_dataset = MammoEvaluation(path=os.path.join(config.PROJECT_ROOT, config.train_data_path), mode=config.prediction_mode)
    elif config.prediction_mode == "testsubmission":
        test_dataset = MammoEvaluation(
            path=os.path.join(config.PROJECT_ROOT, config.train_data_path), mode=config.prediction_mode, ground_truths_path=ground_truths_path, model_name=config.model_name
        )
        # selected_indices = list(range(149))  # Select the first 149 indices for 'testsubmission' mode.
        # test_dataset = Subset(test_dataset, selected_indices)

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=config.num_workers)

    model = torch.load(PREDICTION_MODEL_PATH)
    print(f"Model loaded from {PREDICTION_MODEL_PATH}")
    model = nn.DataParallel(model.module)

    # Check if the base model weights file exists, if not download it
    path = Path(PREDICTION_MODEL_PATH)
    if path.is_file():
        pass
    else:
        url = "https://www.dropbox.com/s/37rtedwwdslz9w6/all_datasets.pth?dl=1"
        response = requests.get(url)
        open(os.path.join(config.PROJECT_ROOT, "test_output/models/weights.pth", "wb")).write(response.content)

        # Decide which mode to process
    if config.prediction_mode == "submission":
        process_submission_mode(test_dataloader, model)
    else:  # 'testsubmission' mode
        process_testsubmission_mode(test_dataloader, model, ground_truths)


if __name__ == "__main__":
    main()
