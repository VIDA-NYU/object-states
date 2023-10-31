# cd ~/ptg/dinov2

# wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth

DATASET=/datasets/annotation_final_imagenet/static_real/state


# python dinov2/eval/knn.py \
#     --config-file dinov2/configs/eval/vits14_reg4_pretrain.yaml \
#     --pretrained-weights dinov2_vits14_reg4_pretrain.pth \
#     --output-dir output/knn \
#     --train-dataset ImageNet:split=TRAIN:root=$DATASET:extra=$DATASET \
#     --val-dataset ImageNet:split=VAL:root=$DATASET:extra=$DATASET


python dinov2/eval/log_regression.py \
    --config-file dinov2/configs/eval/vits14_reg4_pretrain.yaml \
    --pretrained-weights dinov2_vits14_reg4_pretrain.pth \
    --output-dir output/log_regression \
    --train-dataset ImageNet:split=TRAIN:root=$DATASET:extra=$DATASET \
    --val-dataset ImageNet:split=VAL:root=$DATASET:extra=$DATASET\