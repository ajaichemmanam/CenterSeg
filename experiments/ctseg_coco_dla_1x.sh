cd src
# train
python main.py ctseg --exp_id coco_dla_1x --batch_size 20 --master_batch 9 --lr 1.25e-4 --gpus 0,1 --num_workers 4
# python3 main.py ctseg --exp_id coco_dla_1x --batch_size 2 --master_batch -1 --lr 1.25e-4 --gpus -1 --num_workers 0
# test
python test.py ctseg --exp_id coco_dla_1x --keep_res --resume

# Visualize
# python3 demo.py ctseg --exp_id coco_dla_1x --keep_res --resume --demo ../data/coco/val2017
cd ..