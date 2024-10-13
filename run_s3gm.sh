# For KSE
# CUDA_VISIBLE_DEVICES=0 python train.py --data kse --version vx --dims 1 --epochs 300 --batch_size 64 --train_split 0.9 --num_components 1 --num_conditions 1 --attn_resolutions 16 --ch_mult 1 2 4 8 --data_location ./data/kse/KSE_train.npy --verbose 1

# For Kolmogorov flow
# CUDA_VISIBLE_DEVICES=0 python train.py --data kol --version vx --dims 2 --epochs 300 --batch_size 32 --train_split 0.8 --num_components 2 --num_conditions 2 --attn_resolutions 8 16 --ch_mult 1 2 4 8 --data_location ./data/kolmogorov/kolmogorov_flow_train.npy --verbose 1