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
from dataset import MammoEvaluation  # Custom dataset class for mammogram evaluation
from torch.utils.data import DataLoader  # DataLoader class for batch processing
import segmentation_models_multi_tasking as smp  # Library for segmentation models
from tqdm import tqdm  # Library for progress bars
import numpy as np  # Numerical library for array operations
import scipy.stats as stats  # Library for statistical functions
import warnings  # Module to handle warnings
warnings.filterwarnings("ignore")  # Ignoring warnings
import requests  # Module to handle HTTP requests
from pathlib import Path  # Library for handling filesystem paths

"""
Function to parse command line arguments
@returns: The parsed arguments as a config object
"""
def parse_args():
    parser = argparse.ArgumentParser() # Initialize argument parser
    # Adding arguments to the parser :
    parser.add_argument('--data_path', default='data', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='dataset_name', type=str, help='dataset ')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')
    parser.add_argument('--model_save_path', default='test_output/models/unet.pth', type=str, help='path to save the model') # change her
    parser.add_argument('--results_path', default='test_output/test/report.txt', type=str, help='path to save the model')  # change here
    config = parser.parse_args() # Parsing arguments
    return config # Return the parsed arguments as a config object

# Parsing arguments and converting to dictionary (A dictionary in Python is a data structure that stores data in key-value pairs)
config = vars(parse_args())

# Loading the test dataset
test_dataset = MammoEvaluation(path=config['data_path'], dataset=config['dataset'], split='test')

# Creating a DataLoader for batch processing,  
# DataLoader is used to load large datasets or datasets that cannot fit entirely into memory. 
# It helps in efficiently managing memory usage and speeds up the data feeding process to the neural network.
# In machine learning, especially deep learning, models are often trained on a subset of the entire dataset at one time, 
# known as a "batch".
# Batch processing refers to processing data in these small subsets or batches. 
# This approach is more memory-efficient than loading the entire dataset at once, 
# and it can also help in regularization by providing a level of randomness (especially when the data is shuffled for each epoch).

# DataLoader is being configured to handle the dataset created for evaluating a machine learning model.
# The DataLoader takes the test_dataset object (which is an instance of the MammoEvaluation class) and 
# manages how data from this dataset will be fed into the model for evaluation.

# Parameters like shuffle=False and batch_size=1 are specified:
# shuffle=False indicates that the data will not be shuffled before being fed into the model, 
# which is common during evaluation or testing phases.
# batch_size=1 means that the data will be processed one item at a time. In other contexts, 
# especially during training, you might use a larger batch size.
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size =1, num_workers=config['num_workers'])

# Loading the trained model from the given path
model = torch.load(config['model_save_path'])
# Wrapping the model in DataParallel for multi-GPU support
model = nn.DataParallel(model.module)

# Check if the base model weights file exists, if not download it
MODEL_PATH = "test_output/models/weights.pth"
path = Path(MODEL_PATH)
if path.is_file():
    pass
else:
    url = "https://www.dropbox.com/s/37rtedwwdslz9w6/all_datasets.pth?dl=1"
    response = requests.get(url)
    open("test_output/models/weights.pth", "wb").write(response.content)

# Loading the old model for comparison
model_old = torch.load("test_output/models/weights.pth")
model_old = nn.DataParallel(model_old.module)

# Initializing metrics for evaluation
# Intersection over Union (IoU) - Also known as the Jaccard Index, this metric is used in the context of segmentation tasks. 
# It measures the overlap between the predicted segmentation mask and the ground truth mask. Specifically, it calculates 
# the area of overlap between the two masks divided by the area of their union. IoU is a common metric for evaluating the 
# accuracy of object detection and segmentation models because it provides a single number that describes the quality of the 
# model's predictions, taking both false positives and false negatives into account.
IOU = smp.utils.metrics.IoU()

# Precision - This metric calculates the ratio of true positive predictions to the total number of positive predictions made. 
# In other words, it measures the proportion of predicted positives that are actually true positives. High precision indicates 
# that the model returned substantially more relevant results than irrelevant ones. Precision is particularly useful in situations 
# where the cost of a false positive is high.
Precision = smp.utils.metrics.Precision()

# Recall - Recall measures the ratio of true positive predictions to the total actual positives. This metric gives us an 
# indication of the model's ability to find all relevant instances within the dataset. In the context of medical imaging, for example, 
# high recall would mean that the model is able to detect the majority of important features, such as tumors, dense-area or lesions, which is 
# critical for diagnosis.
Recall = smp.utils.metrics.Recall()

# Accuracy - This is one of the most straightforward metrics used in machine learning. It calculates the ratio of the total number 
# of correct predictions (both true positives and true negatives) to the total number of predictions made. While it is easy to 
# understand, accuracy may not always be the best metric to use, especially for imbalanced datasets where one class significantly 
# outnumbers the other.
Accuracy = smp.utils.metrics.Accuracy()

# F-score (or F1 score) - The F-score is the harmonic mean of precision and recall. It is an overall measure of a model's 
# accuracy that considers both the precision and the recall. The F1 score is especially useful when you need to balance precision 
# and recall and is particularly useful in situations with uneven class distributions.
Fscore = smp.utils.metrics.Fscore()

# Dictionary to store the computed metrics for each image.
# This will hold lists of values for different metrics calculated for breast (B) and dense tissue (D) predictions.
test_header_indexs = {
    'B_Prec': [],
    'B_Rec': [],
    'B_F-Sc': [],
    'B_Acc': [],
    'B_IoU': [],
    'D_Prec': [],
    'D_Rec': [],
    'D_F-Sc': [],
    'D_Acc': [],
    'D_IoU': [],
}

# Lists to keep track of the density values computed from the predicted masks.
# These lists will be used later to calculate average and confidence intervals for the predicted densities.
old_dense_values = []
new_dense_values = []
diff_dense_values = []

# Extract the metric names from the dictionary keys to use in the log files' headers.
test_header_names = test_header_indexs.keys()

# Open two files for appending results; one for general metrics, another for density differences.
# 'a+' mode is used to append to the file if it exists or create a new file if it doesn't.
with open(config['results_path'], 'a+') as logs_file, open('test_output/test/report_density_difference.txt', 'a+') as density_file:
    # Reading existing content to avoid overwriting
    logs_file.seek(0)
    existing_content = logs_file.read().strip()
    
    density_file.seek(0)
    dense_existing_content = density_file.read().strip()
    
    # If the files are empty or do not start with the expected headers, write the headers.
    if not existing_content or not existing_content.startswith('Abbreviations') and not dense_existing_content or not existing_content.startswith('Image_ID'):
        # Headers for the logs file, including abbreviations for readability.
        abbreviations = "Abbreviations\nB = Breast\nD = Dense\nPrec = Precision \nRec = Recall\nF-Sc = Fscore\nAcc = Accuracy\n-------------------"
        header = "Image_ID\t\t\t\t" + '\t'.join(test_header_names)
        logs_file.write(abbreviations + '\n')
        logs_file.write(header + '\n')

        # Header for the density differences file.
        dense_header = "Image_ID\t\t\t\tPredicted Density\t\tGround Truth\t\tDifference"
        density_file.write(dense_header + '\n')

        # This loop iterates over each batch of data provided by the test_dataloader. 
        # The test_dataloader yields batches of image IDs (img_id), images, and their corresponding original masks for breast (b_mask_org) 
        # and dense tissue (d_mask_org). The tqdm function is used to wrap the dataloader, which provides a progress bar that shows how 
        # many batches have been processed.
    for (img_id, image, b_mask_org, d_mask_org) in tqdm(test_dataloader):

        # The images and the original masks are moved to the GPU to enable faster computation. 
        # This is necessary because the computations performed by neural networks (like convolutions) 
        # are significantly faster on GPU due to its parallel processing capabilities.
        image = image.cuda()
        b_mask_org = b_mask_org.cuda()
        d_mask_org = d_mask_org.cuda()

        # Here, the model makes predictions based on the input images. The predict function is called on the model, which takes the image
        # tensor and performs a forward pass through the network to output the predicted masks for both breast and dense tissue.
        # Similarly, predictions are made with the 'model_old' which seems to be a previous version of the model, for comparison purposes.
        pred_b_mask, pred_d_mask = model.module.predict(image)
        pred_old_b_mask, pred_old_d_mask = model_old.module.predict(image)

        # After obtaining predictions, the loop iterates over the list of metric names (test_header_names).
        # For each metric name, it checks the prefix ('B_' for breast and 'D_' for dense tissue) to determine whether to calculate the metric
        # for the breast mask or the dense tissue mask. Depending on the metric name, it calls the respective function from the smp.utils.metrics
        # library, passing the predicted mask and the original mask to compute the evaluation metric.
        for metric in test_header_names:

            # The conditionals check the prefix of each metric name to determine which metric to calculate.
            # If the metric name starts with 'B_P', it calculates the precision for the breast mask prediction, and so on for other metrics.
            # Each metric function returns a tensor, which is converted to a Python float with the .item() method and rounded to three decimal places.
            if metric.startswith('B_P'):
                value = Precision(pred_b_mask, b_mask_org)
            elif metric.startswith('B_R'):
                value = Recall(pred_b_mask, b_mask_org)
            elif metric.startswith('B_F'):
                value = Fscore(pred_b_mask, b_mask_org)
            elif metric.startswith('B_A'):
                value = Accuracy(pred_b_mask, b_mask_org)
            elif metric.startswith('B_I'):
                value = IOU(pred_b_mask, b_mask_org)
            elif metric.startswith('D_P'):
                value = Precision(pred_d_mask, d_mask_org)
            elif metric.startswith('D_R'):
                value = Recall(pred_d_mask, d_mask_org)
            elif metric.startswith('D_F'):
                value = Fscore(pred_d_mask, d_mask_org)
            elif metric.startswith('D_A'):
                value = Accuracy(pred_d_mask, d_mask_org)
            else:
                value = IOU(pred_d_mask, d_mask_org)

            # The computed metric value is then added to a list in the test_header_indexs dictionary corresponding to that metric.
            # This allows all the metric values to be stored and later accessed for analysis or logging purposes.
            value = round(value.item(), 3) # Round the metric value to 3 decimal places.
            test_header_indexs[metric].append(value) # Append the value to the corresponding list in the dictionary.
            #print(test_header_indexs)

        # Processing of predicted masks to extract area values for breast and dense tissues. 
        # The masks are moved from GPU to CPU, converted to numpy arrays, and reshaped to two-dimensional arrays.
        # Area calculations are then performed for both current and old model predictions.
        # The area is calculated by counting the number of pixels labeled as '1' (indicative of tissue presence).
        # The density percentage is computed by dividing the dense area by the breast area.
        # The 'diff' variable captures the absolute difference in density percentages between the current and old model predictions.
        # This process is repeated for both the new and old model predictions.
        # The calculated values are appended to their respective lists for later analysis.

        pred_b_mask = pred_b_mask[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]
        pred_d_mask = pred_d_mask[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]
        pred_old_b_mask = pred_old_b_mask[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]
        pred_old_d_mask = pred_old_d_mask[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]

        # Calculation of the breast and dense tissue areas for the current model predictions.
        # The area is determined by counting the number of pixels that are classified as tissue (value == 1).
        breast_area = np.sum(np.array(pred_b_mask) == 1)
        dense_area = np.sum(np.array(pred_d_mask) == 1)

        # Calculation of the density percentage for the current model.
        # It's calculated by dividing the dense tissue area by the breast area and multiplying by 100 to get a percentage.
        # The result is rounded to three decimal places for precision.
        new_density = round(((dense_area / breast_area) * 100), 3)

        # Similar calculations are performed for the old model predictions.
        old_breast_area = np.sum(np.array(pred_old_b_mask) == 1)
        old_dense_area = np.sum(np.array(pred_old_d_mask) == 1)
        old_density = round(((old_dense_area / old_breast_area) * 100), 3)

        # Calculation of the absolute difference in density percentages between the new and old model predictions.
        # This difference gives insight into the performance change or improvement of the new model over the old model.
        diff = round(abs(new_density - old_density), 3)

        # Appending the calculated density values for both models and their difference to respective lists.
        # These lists store the density values for each image in the dataset for later analysis or comparison.
        old_dense_values.append(old_density)
        new_dense_values.append(new_density)
        diff_dense_values.append(diff)

        # Compilation of the latest metric values for the current image.
        # This extracts the most recent value for each metric from the test_header_indexs dictionary.
        metric_values = [test_header_indexs[metric][-1] for metric in test_header_names]
        row = '{}\t{}'.format(img_id[0], '\t'.join(map(str, metric_values)))
        print(row, file=logs_file)

        # Formatting and writing the metric values to the logs file.
        # Each line in the logs file corresponds to a single image and its associated metric values.
        dense_row = '{}\t{}\t\t\t{}\t\t\t{}'.format(img_id[0], new_density, old_density, diff)
        print(dense_row, file=density_file)

    # Definition of the mean_cal function. It calculates and returns the mean of the provided data,
    # rounding the result to three decimal places for precision and readability.
    def mean_cal(data):
        return round(np.mean(data), 3)
    
    # Definition of the average_count function. It calculates the confidence interval for the mean of the provided data.
    # It uses a t-distribution (appropriate for small sample sizes) with a 95% confidence level.
    # 'stats.t.interval' calculates the interval, 'len(data)-1' gives the degrees of freedom,
    # 'loc=mean' specifies the mean around which to calculate the interval, and 'scale=stats.sem(data)' uses the standard error of the mean.
    # The function returns the width of the confidence interval, rounded to three decimal places.
    def average_count(data):
        mean = mean_cal(data)
        ci_min, ci_max = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
        return round((ci_max - ci_min), 3)
    
    # Construction of a string to log the average values of the metrics.
    # It uses a list comprehension to calculate the mean of each metric in test_header_indexs,
    # formats each mean to three decimal places, and joins them with tabs.
    avg_ci_row = '\nAverage\t\t\t\t\t' + '\t'.join([f'{mean_cal(test_header_indexs[metric]):.3f}' for metric in test_header_names])

    # Construction of a string to log the confidence intervals for the metrics,
    # following a similar process as for the averages.
    avg_ci_row += '\nConfidance Interval\t\t' + '\t'.join([f'{average_count(test_header_indexs[metric]):.3f}' for metric in test_header_names])
    
    # Writing the average and confidence interval data to the logs file.
    print(avg_ci_row, file=logs_file)

    # Construction and logging of similar data for the density values. 
    # This includes the average and confidence interval for old and new density values, as well as their differences.
    dense_avg_ci_row = '\nAverage\t\t\t\t\t{:.3f}\t\t\t{:.3f}\t\t\t{:.3f}'.format(
        np.mean(old_dense_values), np.mean(new_dense_values), np.mean(diff_dense_values)
    )
    dense_avg_ci_row += '\nConfidance Interval\t\t{:.3f}\t\t\t{:.3f}\t\t\t{:.3f}'.format(
        average_count(old_dense_values), average_count(new_dense_values), average_count(diff_dense_values)
    )

    #Lets print and have some fun!
    print(dense_avg_ci_row, file=density_file)


