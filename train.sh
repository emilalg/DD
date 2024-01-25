DIR="$( cd "$( dirname "$0" )" && pwd )"

python scr/train.py --data_path ${DIR}/data/ --dataset test1 --logs_file_path test_output/logs/test1.txt --model_save_path test_output/models/test1.pth --num_epochs 5