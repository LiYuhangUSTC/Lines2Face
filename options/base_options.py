import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        ## experiment
        self.parser.add_argument("--input_dir", help="path to folder containing images")
        self.parser.add_argument("--mode", required=True, choices=["train", "test"])
        self.parser.add_argument("--train_stage", default="stage_one", choices=["stage_one", "stage_two", "stage_three"])
        self.parser.add_argument("--finetune", dest="finetune", action="store_true", help="finetune model from a restored model") 
        self.parser.set_defaults(finetune=False)
        self.parser.add_argument("--output_dir", required=True, help="where to put output files")
        self.parser.add_argument("--seed", type=int)
        self.parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

        ## training setting
        self.parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
        self.parser.add_argument("--max_epochs", type=int, default=100, help="number of training epochs")
        self.parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")

        ## architecture selection
        self.parser.add_argument("--gen_resgan_arch", default="0", type=int, help="resgan generator archtecture type")
        self.parser.add_argument("--discriminator", default="conv", choices=["res", "ir", "conv", "mru", "sa", "sa_I", "resgan", "double", "triple", "quadruple", "pix2pixhd"])
        self.parser.add_argument("--input_type", default="df", choices=["edge", "df", "hed", "vg"])

        
        self.parser.add_argument("--use_dropout", dest="dropout", action="store_true", help="use pretrained vgg model for perceptual loss")
        self.parser.set_defaults(dropout=False)
        self.parser.add_argument("--use_vgg", dest="vgg", action="store_true", help="use pretrained vgg model for perceptual loss")
        self.parser.set_defaults(vgg=False)


        self.parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
        self.parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
        self.parser.add_argument("--num_unet", type=int, default=10, help="number of u-connection layers, used only when generator is encoder-decoder")

        self.parser.add_argument("--channel_fac", default=16, type=int, help="faction of channel in self attention modual. Set to large to save GPU memory")
        self.parser.add_argument("--enc_atten", type=str, default="FTFFF")
        self.parser.add_argument("--dec_atten", type=str, default="FFFTF")
        self.parser.add_argument("--use_attention", dest="attention", action="store_true", help="finetune model from a restored model") 
        self.parser.set_defaults(attention=False)

        
        self.parser.add_argument("--no_sn", dest="sn", action="store_false", help="do not use spectral normalization")
        self.parser.set_defaults(sn=True)
        self.parser.add_argument("--no_fm", dest="fm", action="store_false", help="do not use feature matching loss")
        self.parser.set_defaults(fm=True)
        self.parser.add_argument("--no_style_loss", dest="style_loss", action="store_false", help="do not use style loss")
        self.parser.set_defaults(style_loss=True)

        self.parser.add_argument("--num_residual_blocks", type=int, default=8, help="number of residual blocks in resgan generator")
        self.parser.add_argument("--num_feature_matching", type=int, default=3, help="number of layers in feature matching loss, count from the last layer of the discriminator")
        self.parser.add_argument("--num_style_loss", type=int, default=3, help="number of layers in style loss, count from the last layer of the discriminator")
        self.parser.add_argument("--num_vgg_class", type=int, default=1000, help="number of class of pretrained vgg network")

        ## preprocessing image
        self.parser.add_argument("--scale_size", type=int, default=530, help="scale images to this size before cropping to 256x256")
        self.parser.add_argument("--target_size", type=int, default=512, help="scale images to this size before cropping to 256x256")
        self.parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
        self.parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
        self.parser.set_defaults(flip=True)
        self.parser.add_argument("--random_crop", dest="random_crop", action="store_true", help="crop images randomly")
        self.parser.set_defaults(random_crop=False)
        self.parser.add_argument("--monochrome", dest="monochrome", action="store_true", help="convert image from rgb to gray")
        self.parser.set_defaults(monochrome=False)

        self.parser.add_argument("--df_threshold", type=float, default=0.5, help="the nomalizaiton value of distance fields")
        self.parser.add_argument("--df_norm", default="value", choices=["max", "value"])
        self.parser.add_argument("--df_norm_value", type=float, default=64.0, help="the nomalizaiton value of distance fields")


        ## input/output setting
        self.parser.add_argument("--load_image", dest="load_tfrecord", action="store_false", help="if true, read dataset from TFRecord, otherwise from images")
        self.parser.set_defaults(load_tfrecord=True)
        self.parser.add_argument("--num_examples", required=True, type=int, help="number of training/testing examples in TFRecords. required, since TFRecords do not have metadata")
        self.parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")

        ## multiple gpu setting
        self.parser.add_argument("--num_gpus", type=int, default=4, help="number of GPUs used for training")
        self.parser.add_argument("--num_gpus_per_tower", type=int, default=2, help="number of GPUs per tower used for training")

        ## export options
        self.parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])        

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('--------- End of Options ----------------')
        return self.opt

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        #util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
