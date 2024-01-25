import argparse
import torch
import torch.nn as nn
from dataset import MammoEvaluation
from torch.utils.data import DataLoader
import segmentation_models_multi_tasking as smp
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
import requests 
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='dataset_name', type=str, help='dataset ')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')
    parser.add_argument('--model_save_path', default='test_output/models/unet.pth', type=str, help='path to save the model') # change her
    parser.add_argument('--results_path', default='test_output/test/report.txt', type=str, help='path to save the model')  # change here
    config = parser.parse_args()
    return config

config = vars(parse_args())

test_dataset = MammoEvaluation(path=config['data_path'], dataset=config['dataset'], split='test')
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size =1, num_workers=config['num_workers'])

model = torch.load(config['model_save_path'])
model = nn.DataParallel(model.module)
# Base Model
MODEL_PATH = "test_output/models/weights.pth"
path = Path(MODEL_PATH)
if path.is_file():
    pass
else:
    url = "https://www.dropbox.com/s/37rtedwwdslz9w6/all_datasets.pth?dl=1"
    response = requests.get(url)
    open("test_output/models/weights.pth", "wb").write(response.content)

model_old = torch.load("test_output/models/weights.pth")
model_old = nn.DataParallel(model_old.module)

IOU = smp.utils.metrics.IoU()
Precision = smp.utils.metrics.Precision()
Recall = smp.utils.metrics.Recall()
Accuracy = smp.utils.metrics.Accuracy()
Fscore = smp.utils.metrics.Fscore()

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

old_dense_values = []
new_dense_values = []
diff_dense_values = []

test_header_names = test_header_indexs.keys()

with open(config['results_path'], 'a+') as logs_file, open('test_output/test/report_density_difference.txt', 'a+') as density_file:
    logs_file.seek(0)
    existing_content = logs_file.read().strip()
    
    density_file.seek(0)
    dense_existing_content = density_file.read().strip()
    
    if not existing_content or not existing_content.startswith('Abbreviations') and not dense_existing_content or not existing_content.startswith('Image_ID'):
        abbreviations = "Abbreviations\nB = Breast\nD = Dense\nPrec = Precision \nRec = Recall\nF-Sc = Fscore\nAcc = Accuracy\n-------------------"
        header = "Image_ID\t\t\t\t" + '\t'.join(test_header_names)
        logs_file.write(abbreviations + '\n')
        logs_file.write(header + '\n')
        dense_header = "Image_ID\t\t\t\tPredicted Density\t\tGround Truth\t\tDifference"
        density_file.write(dense_header + '\n')

    for (img_id, image, b_mask_org, d_mask_org) in tqdm(test_dataloader):
        image = image.cuda()
        b_mask_org = b_mask_org.cuda()
        d_mask_org = d_mask_org.cuda()

        pred_b_mask, pred_d_mask = model.module.predict(image)
        pred_old_b_mask, pred_old_d_mask = model_old.module.predict(image)

        for metric in test_header_names:
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

            value = round(value.item(), 3)
            test_header_indexs[metric].append(value)
            #print(test_header_indexs)

        pred_b_mask = pred_b_mask[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]
        pred_d_mask = pred_d_mask[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]
        pred_old_b_mask = pred_old_b_mask[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]
        pred_old_d_mask = pred_old_d_mask[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]
        breast_area = np.sum(np.array(pred_b_mask) == 1)
        dense_area = np.sum(np.array(pred_d_mask) == 1)
        new_density = round(((dense_area / breast_area) * 100), 3)
        old_breast_area = np.sum(np.array(pred_old_b_mask) == 1)
        old_dense_area = np.sum(np.array(pred_old_d_mask) == 1)
        old_density = round(((old_dense_area / old_breast_area) * 100), 3)
        diff = round(abs(new_density - old_density), 3)

        old_dense_values.append(old_density)
        new_dense_values.append(new_density)
        diff_dense_values.append(diff)

        metric_values = [test_header_indexs[metric][-1] for metric in test_header_names]
        row = '{}\t{}'.format(img_id[0], '\t'.join(map(str, metric_values)))
        print(row, file=logs_file)

        dense_row = '{}\t{}\t\t\t{}\t\t\t{}'.format(img_id[0], new_density, old_density, diff)
        print(dense_row, file=density_file)

    def mean_cal(data):
        return round(np.mean(data), 3)
    
    def average_count(data):
        mean = mean_cal(data)
        ci_min, ci_max = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
        return round((ci_max - ci_min), 3)
    
    avg_ci_row = '\nAverage\t\t\t\t\t' + '\t'.join([f'{mean_cal(test_header_indexs[metric]):.3f}' for metric in test_header_names])
    avg_ci_row += '\nConfidance Interval\t\t' + '\t'.join([f'{average_count(test_header_indexs[metric]):.3f}' for metric in test_header_names])
    print(avg_ci_row, file=logs_file)

    dense_avg_ci_row = '\nAverage\t\t\t\t\t{:.3f}\t\t\t{:.3f}\t\t\t{:.3f}'.format(
        np.mean(old_dense_values), np.mean(new_dense_values), np.mean(diff_dense_values)
    )
    dense_avg_ci_row += '\nConfidance Interval\t\t{:.3f}\t\t\t{:.3f}\t\t\t{:.3f}'.format(
        average_count(old_dense_values), average_count(new_dense_values), average_count(diff_dense_values)
    )
    print(dense_avg_ci_row, file=density_file)