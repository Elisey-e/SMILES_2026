source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python3 validate.py --data_dir ./data --batch_size 64 --n_batches 128 --output results.json # --seed 1256
