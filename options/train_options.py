from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        ## save and display
        self.parser.add_argument("--summary_freq", type=int, default=20, help="update summaries every summary_freq steps")
        self.parser.add_argument("--progress_freq", type=int, default=20, help="display progress every progress_freq steps")
        self.parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
        self.parser.add_argument("--display_freq", type=int, default=50, help="write current training images every display_freq steps")
        self.parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
        self.parser.add_argument("--evaluate_freq", type=int, default=5000, help="evaluate training data every save_freq steps, 0 to disable")
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

         ## training hyper-parameters
        self.parser.add_argument("--lr_gen", type=float, default=0.0001, help="initial learning rate for adam")
        self.parser.add_argument("--lr_discrim", type=float, default=0.0004, help="initial learning rate for adam")
        self.parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")

        self.parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
        self.parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
        self.parser.add_argument("--fm_weight", type=float, default=1.0, help="weight on feature matching term for generator gradient")
        self.parser.add_argument("--style_weight", type=float, default=1.0, help="weight on style loss term for generator gradient")

        self.parser.add_argument("--l1_weight_decay_steps", type=int, default=10000, help="learning rate decay steps for l1 loss weight") # count by step, should count by epoch
        self.parser.add_argument("--l1_weight_decay_factor", type=float, default=0.9, help="learning rate decay factor for l1 loss weight")

        self.parser.add_argument("--lr_decay_steps_D", type=int, default=10000, help="learning rate decay steps for discriminator") # count by step, should count by epoch
        self.parser.add_argument("--lr_decay_steps_G", type=int, default=10000, help="learning rate decay steps for generator")
        self.parser.add_argument("--lr_decay_factor_D", type=float, default=0.1, help="learning rate decay factor for discriminator")
        self.parser.add_argument("--lr_decay_factor_G", type=float, default=0.1, help="learning rate decay factor for generator")

        self.isTrain = True
