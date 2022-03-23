import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_common_args(parser):
    """
    common args shared in training and testing
    """
    ### related to data ###
    # data/label directory path 
    parser.add_argument('--img_dir', type=str,
                        default="/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/",
                        help='the directory path of the dataset, the .txt file is in the corresponding root directory!!!')
    parser.add_argument('--label_dir', type=str,
                        default="/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/",
                        help='the directory path of the labels,  the .txt file is in the corresponding root directory!!!')
    # parser.add_argument('--img_dir', type=str,
    #                     default="/home/zj/data/potsdam/data_split/",
    #                     help='the directory path of the dataset, the .txt file is in the corresponding root directory!!!')
    # parser.add_argument('--label_dir', type=str,
    #                     default="/home/zj/data/potsdam/label_split/",
    #                     help='the directory path of the labels,  the .txt file is in the corresponding root directory!!!')
    # select model and dataset
    # parser.add_argument('--model', type=str, default='AttUResNeXt_class_v2', help='used in model_entry.py to select model')
    parser.add_argument('--model', type=str, default='UResNeXt', help='used in model_entry.py to select model')
    parser.add_argument('--dataset', type=str, default='Gid', help='used in data_entry.py to select dataset')
    # dataset
    parser.add_argument('--num_classes', type=int, default=16, help='number of classes of the dataset')
    # model
    parser.add_argument('--load_not_strict',default=True,type=str2bool, help='allow to load only common state dicts')
    # parser.add_argument('--load_model_path', type=str, default='checkpoints/gid_noverlap/AttUResNeXt_class_v2/12/best/AttUResNeXt_class_v2_088.pth.tar', help="model path for pretrain or test, use '' for no pretrain")
    # parser.add_argument('--load_model_path', type=str, default='checkpoints/postdam/AttUResNeXt_class_v2/07/best/AttUResNeXt_class_v2_070.pth.tar', help="model path for pretrain or test, use '' for no pretrain")
    parser.add_argument('--load_model_path', type=str, default='checkpoints/gid_noverlap/UResNeXt/12/best/UResNeXt_086.pth.tar', help="model path for pretrain or test, use '' for no pretrain")
    parser.add_argument('--is_pretrain', type=str2bool,  default=True, help='use saved model for pretrain or checkpoints')
    parser.add_argument('--is_pretrained', type=str2bool, default=True, help='whether using pretrained weight of the backbone, true for using lower lr for the backbone')
    # for reproduction
    parser.add_argument('--seed', type=int,
                        default=0,
                        help='for reproduction')
    # for parallel training
    parser.add_argument('--is_gpu',default=True, type=bool, help='True for running in gpu, false for running in cpu')
    parser.add_argument('--gpus', nargs='+',default=[0], type=int)
    parser.add_argument('--gpu_id',default="0", type=str,help="designate gpu for training, should be a str like '1,7'")




    # parser.add_argument('--save_prefix', type=str, default='pref', help='some comment for model or test result dir')

    # parser.add_argument('--val_list', type=str, default='/data/dataset1/list/base/val.txt',
    #                     help='val list in train, test list path in test')
    return parser


### for trainer ###

def parse_train_args(parser):
    """
    Description:
        parser for training
    Params:
        parser -- common parser
    """
    parser = parse_common_args(parser)
    ### for model ###
    parser.add_argument('--freeze_model', type=str, default='UResNeXt', help='the model used for freezing dict')
    parser.add_argument('--is_freeze', type=bool, default=False, help='whether freezing the weight of the particular layer')

    ### for optimizer ###
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                        metavar='W', help='weight decay')

    ### for dataset ###
    parser.add_argument('--res', type=int, default=800, help='for selecting resolution of gid dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='for batch size of val dataset')
    parser.add_argument('--num_workers', type=int, default=8, help="for prefetching data")
    parser.add_argument('--epochs', type=int, default=90)

    ### lr scheduler ###
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau', help='select lr scheduler')
    # ReduceLROnPlateau
    parser.add_argument('--patience', type=int, default=8, help='patience for learning rate dacay of ReducerOnPlateau')
    parser.add_argument('--ropfactor', type=float, default=0.1, help='reducing factor for reduce on plateau')
    # CosineAnnealingWarmRestarts -- the lr of this lr_scheduler is always set to 0.01
    parser.add_argument('--T_0', type=int, default=10, help='number of iterations for the first restart')
    parser.add_argument('--T_mult', type=int, default=2, help='A factor increases T_i after a restart')
    # ConsineOneCircle
    parser.add_argument('--lrf', type=float, default=0.01, help='weight between primitive lr and cosinelr')

    ### for saving ###
    parser.add_argument('--model_dir', type=str, default='checkpoints/gid', help='directory to save the model and other args and training statistics')
    parser.add_argument('--log_dir', type=str, default='./logs/gid', help='logs used for summarywriter')
    parser.add_argument('--save_suffix', type=str, default='',
                        help='as a supplementary information for the dir_name, appended after the exp_id, \
                              in the form of "_...", or '' for no suffix')
    parser.add_argument('--exp_id', type=int, default=0, help='id of the experiment of the corresponding model')
    parser.add_argument('--model_epoch', type=int, default=10, help='save checkpoints every model_epoch')
    parser.add_argument('--current_epoch', type=int, default=0, help='the current epoch, obtained from the checkpoints.pth.tar file')
    

    # for loss
    parser.add_argument('--loss', type=str, default='ce', help='choose the loss of the training process')
    # for monitoring
    parser.add_argument('--print_freq', type=int, default=100, help='frequency to print the training message')

    # for more training information
    parser.add_argument('--exp_information', type=str,default='train with res 800', help='information of various settings for the experiment')
    
    return parser

def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz', default=True, help='save viz result in eval or not')
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--viz_freq', type=int, default=40, help='visiualize the result every viz_freq batch')
    # for dataset
    parser.add_argument('--res', type=int, default=320, help='for selecting resolution of the tested dataset')
    parser.add_argument('--batch_size', type=int, default=2)
    # for testing method
    parser.add_argument('--is_tta', type=str2bool, default=True, help='leave blank, auto generated')
    
    return parser

### prepare args ###
def prepare_train_args():
    args = get_train_args()
    get_train_model_dir(args)
    save_args(args, args.model_dir)
    return args

def prepare_test_args():
    args = get_test_args()
    get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args

### get args ###
def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args

def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args

### save args ###
def save_args(args, save_dir):
    """
    Description:
        save args as a log for the corresponding experiment
    """
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))

### prepare directory for saving results and models ###
def get_train_model_dir(args):
    args.model_dir = os.path.join(args.model_dir, args.model, "{:02d}".format(args.exp_id) + args.save_suffix)
    if not os.path.exists(args.model_dir):
        os.system('mkdir -p ' + args.model_dir) # note there is a space after '-p'
        model_dir = os.path.join(args.model_dir, 'best')
        os.system('mkdir -p ' + model_dir)
        

def get_test_result_dir(args):
    # get rid of extension
    ext = os.path.basename(args.load_model_path).split('.')[-1]
    args.model_dir = args.load_model_path.replace(ext, '')
    # the directory to save the test result
    result_dir = os.path.join(args.model_dir + '_test_info')
    if not os.path.exists(result_dir):
        os.system('mkdir -p ' + result_dir)
    args.result_dir = result_dir


def get_checkpoint_path(args):
    # get rid of the name of the saved model file
    file_name = os.path.basename(args.load_model_path)
    checkpoint_dir = args.model_dir.replace(file_name, '')
    checkpoint_path = os.path.join(checkpoint_dir, 'state', file_name)
    return checkpoint_path
