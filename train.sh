DIR="$( cd "$( dirname "$0" )" && pwd )"

python scr/train.py --data_path ${DIR}/data/ --dataset test1 --logs_file_path test_output/logs/test1.txt --model_save_path test_output/models/test1.pth --num_epochs 5

#example evaluate command
# python scr/evaluate.py --data_path ${DIR}/data/ --dataset dataset_name --test_batch_size 1 --num_epochs 10 --num_workers 0 --loss_fucntion LOSS --model_save_path test_output/models/test1.pth --results_path test_output/evaluation/dataset_name.txt