## Deepdensity  
# UEF Breast Cancer Group - Challenge
>This repository is a baseline model for estimating breast density from mammograms.



#
<figure>
  <img src="images/MTLSegNet (2).png" alt="Image Caption">
  <figcaption>An advanced architecture for accurate mammogram segmentation. Its encoder extracts imaging features, the bottleneck enhances spatial information, and task-specific decoders segment breast area and dense tissues. Our modified loss function ensures optimal performance. Predicted segmentations are overlaid on the mammogram, with red contour for breast area and solid green for fibroglandular tissues. MTLSegNet revolutionizes mammogram analysis, enabling improved medical diagnoses.
  <a href="https://www.nature.com/articles/s41598-022-16141-2">Reference</a>
  </figcaption>
</figure>

#
## 1. Envoirnment
- Conda
- Python>=3.8
- CPU or NVIDIA GPU + CUDA CuDNN 
    -  (CUDA Version: 11.7 & Model: Quadro P1000)

Install python packages
```
1. git clone https://github.com/uefcancer/Deepdensity.git
2. cd Deepdensity
3. pip install -r requirements.txt
```
#
## 2. Dataset Structure

```
  data
    ├──dataset_name
            ├──train
                ├── breast_mask
                    ├── 00000_train_1.jpg
                    ├── 00001_train_3.jpg
                    └── ...
                ├── input_image
                    ├── 00000_train_1.jpg
                    ├── 00001_train_3.jpg
                    └── ...
                ├── dense_mask
                    ├── 00000_train_1.jpg
                    ├── 00001_train_3.jpg
                    └── ...

            ├──val
                ├── breast_mask
                    ├── 00000_val_1.jpg
                    ├── 00001_val_3.jpg
                    └── ...
                ├── input_image
                    ├── 00000_val_1.jpg
                    ├── 00001_val_3.jpg
                    └── ...
                ├── dense_mask
                    ├── 00000_val_1.jpg
                    ├── 00001_val_3.jpg
                    └── ...
            

  ```
> Data will be provided in a zip file. Access data by clicking [here]().


#
## 3. train.py


```
python scr/train.py --data_path /path/to/data --dataset dataset_name --logs_file_path test_output/logs/abc.txt --model_save_path test_output/models/abc.pth --num_epochs 5
```
- To store the output files in the desired format, create the following folders:
     - Log file: `test_output/logs/abc.txt`
     - Model file: `test_output/models/abc.pth`
    
- Replace `abc` with the desired name for your log and model files. This format ensures that the logs and models are saved in separate folders for better organization.

#
## 4. prediction.py


```
python scr/predictions.py --data_path data --dataset dataset_name --results_path test_output/logs/results.txt --model_path test_output/models/abc.pth  --density_compare test_output/logs/density_comparision.txt
```
- To estimate the breast area, dense area and percentage density
  
- To store the output files in the desired format, create the following folders:
     - Result file for Test Data: `test_output/logs/results.txt`
     - Density file (Image Wise): `test_output/models/density_comparision.pth`
    
- These files have Image wise output. 
    - `Result File` contain `Precision`, `Recall`, `Fscore`, `Accuracy`, `IoU`
    - `Density file` contain `Predicted Density`, `Ground Truth (Baseline Density)`, `Absolute Mean Difference of Densities`


## 4. evaluate.py


- To report segmentation metrics of breast and dense tissue segmentations
- 

#
## 5. Hyper paramter information

| Hyperparameters | Search hyperparameters  | Optimal values |
| -------- | -------- | -------- |
| Training optimizers   | (Stochastic gradient descent, Adam, RMSprop)    | Adam  |
| Learning rate schedulers   | (StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR)   | ReduceLROnPlateau   |
| Initial learning rate   | (le-1, le-2, le-3, le-4, le-5)   | le-3   |
| Loss functions   | (BCEwithlogits, Dice, Tversky, focal Tversky)   | focal Tversky   |

>Introducing our meticulously honed parameter values. ! But that's not all – we believe in the power of collaboration. We warmly invite you to bring your own hyperparameters values, unlocking the potential for even more accurate and groundbreaking models.

#
## 6. Citation
If our work has made a positive impact on your research endeavors, we kindly request you to acknowledge our contribution by citing our paper.

    @article{gudhe2022area,
      title={Area-based breast percentage density estimation in mammograms using weight-adaptive multitask learning},
      author={Gudhe, Naga Raju and Behravan, Hamid and Sudah, Mazen and Okuma, Hidemi and Vanninen, Ritva and Kosma, Veli-Matti and Mannermaa, Arto},
      journal={Scientific reports},
      volume={12},
      number={1},
      pages={12060},
      year={2022},
      publisher={Nature Publishing Group UK London}
    }

#
## 7. Contact
In case you run into any obstacles along the way, don't hesitate to raise an issue! We're dedicated to providing you with full support and resolving any difficulties you may encounter.

Stay Connected:

    Team
        - Hamid Behravan, PhD. (hamid.behravan@uef.fi)
        - Raju Gudhe, MSc. (raju.gudhe@uef.fi)
        - Arto Mannermaa, professor (arto.mannermaa@uef.fi)


## 8. Acknowledgements
Grateful to the open-source projects and their visionary authors for their generous contributions that inspired and empowered our project. 
    
- [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch)
