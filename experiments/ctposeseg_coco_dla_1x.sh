cd src
# train
python3 main.py ctposeseg --exp_id coco_dla_1x --dataset coco_hp --batch_size 20 --master_batch 9 --lr 1.25e-4 --gpus 0,1 --num_workers 0
# python3 main.py ctposeseg --exp_id coco_dla_1x --batch_size 2 --master_batch -1 --lr 1.25e-4 --gpus -1 --num_workers 0
# test
python3 test.py ctposeseg --exp_id coco_dla_1x --dataset coco_hp --keep_res --resume

# Visualize
python3 demo.py ctposeseg --exp_id coco_dla_1x --dataset coco_hp --keep_res --resume --demo ../data/coco/val2017
cd ..