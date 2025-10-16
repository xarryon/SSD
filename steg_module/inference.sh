export CUDA_HOME=/usr/local/cuda-11.2

python inference.py
# wait
# accelerate launch train_2_sum.py --config configs/config_2.yaml
# wait
# accelerate launch train_2.py --config configs/config_3.yaml