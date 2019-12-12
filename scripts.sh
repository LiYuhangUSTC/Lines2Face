# test
python test.py \
--mode test \
--checkpoint checkpoint \
--output_dir /gdata/liyh/data/CelebA-HD/output/release_quadruple \
--max_epochs 200 \
--input_dir /gdata/liyh/data/CelebA-HD/data/tfrecord/test  \
--batch_size 1 \
--num_examples 6000 \
--discriminator quadruple \
--input_type df \
--scale_size 256 \
--target_size 256 \
--use_attention