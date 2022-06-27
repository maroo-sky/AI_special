#HW03 Quantization
##ResNet18

    DATA_DIR = /home/s1_u1/projects/quantization/dataset/CIFAR10
    OUT_DIR = /home/s1_u1/projects/quantization/checkpoints/resnet18.pt
###training
    python projects/quantization/grid_main_cifar10.py 
    --dataset DATA_DIR
    --out_dir OUT_DIR
    --gpu_id 0 
    --optimizer sgd 
    --weight_decay 4e-5 
    --num_epoch 120 
    --lr_rate 2e-2 
    --train_batch_size 256 
    --seed 100 
    --do_train 
    --do_eval 
    --write
    
Accuracy: 92.04%

##Quantized ResNet18 (DQ)


##Quantized ResNet18 (QAT)

###training
    python projects/quantization/grid_mian_qcifar10.py 
    --dataset DATA_DIR
    --out_dir OUT_DIR
    --gpu_id 0 
    --num_bits 16
    --optimizer sgd 
    --weight_decay 2e-5
    --num_epoch 120 
    --lr_rate 1e-1 
    --train_batch_size 256 
    --seed 100 
    --do_train 
    --do_eval
    
Accuracy: 91.51%