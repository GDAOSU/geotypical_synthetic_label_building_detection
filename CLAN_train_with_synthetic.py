import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import json
import shutil

from model.CLAN_G import Res_Deeplab
from model.HRNetv2_G import HRNetV2
from model.HRNetv2OCR_G import HRNetV2OCR
from model.CLAN_D import FCDiscriminator
from tqdm import tqdm

from utils.loss import CrossEntropy2d
from utils.loss import WeightedBCEWithLogitsLoss

from CLAN_iou import fast_hist, label_mapping, per_class_iu

from dataset.cityengine_dataset import CityEngineDataSet
from dataset.target_dataset import TargetDataSet
from torch.utils.tensorboard import SummaryWriter

IMG_MEAN = np.array((128, 128, 128), dtype=np.float32)

MODEL = 'HRNetV2OCR'
BATCH_SIZE = 8
ITER_SIZE = 1
NUM_WORKERS = 4

IGNORE_LABEL = 255


MOMENTUM = 0.9
NUM_CLASSES = 2
RESTORE_FROM = './model/DeepLab_resnet_pretrained_init-f81d91e8.pth'
RESTORE_FROM = ''
# RESTORE_FROM = './snapshots/HRNetV2OCR_OSU_RandBig_OSU2/GTA5_62000.pth' #For retrain
# RESTORE_FROM_D = './snapshots/HRNetV2OCR_OSU_RandBig_OSU2/GTA5_62000.pth' #For retrain

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './snapshots/'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 0.005 #2.5e-4
LEARNING_RATE_D = 0.005
NUM_STEPS = 50000
NUM_STEPS_STOP = 50000  # Use damping instead of early stopping
PREHEAT_STEPS = 5000 # int(NUM_STEPS_STOP/100)
POWER = 0.9
RANDOM_SEED = 1234

#  'OSMOSU2DR_building'     Columbus_syn
#  'OSU_building'           Columbus
#  'tyrol-wAll'             Tyrol
#  'OSMOSU2_1_DR'           Columbus_syn no domain randomization
#  'DSTL_building'          DSTL
#  'chicagoAll'             Chicago
#  'chicago_syn'            Chicago_syn
#  'austinAll'              Ausstin
#  'SynRS3D'                2024 New data from other researchers. Tokyo university.
#  'Syntheworld'            2024 New data 2 from other researchers

FOLDER = 'building_dataset'
SOURCE = 'DSTL_building' # austinAll tyrol-wAll DSTL_building
ADAPTER = 'Syntheworld' # SynRS3D    Syntheworld       OSMOSU2DR_building chicago_syn
TARGET = 'chicagoAll' # OSU_building chicagoAll
SET = 'train'
EXPERIENCE_NAME = '16'


DATA_DIRECTORY_ADAPTER = f'./{FOLDER}/{ADAPTER}'
DATA_LIST_PATH_ADAPTER = f'./{FOLDER}/{ADAPTER}/train.txt'


def readInfo(devkit_dir):
    with open(osp.join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = int(info['classes'])
    name_classes = np.array(info['label'], dtype=str)
    mapping = np.array(info['label2train'], dtype=int)
    return num_classes, name_classes, mapping

if SOURCE:
    INPUT_SIZE_SOURCE = '512,512'
    DATA_DIRECTORY = f'./{FOLDER}/{SOURCE}'
    DATA_LIST_PATH = f'./{FOLDER}/{SOURCE}/train.txt'
    DATA_VAL_LIST_PATH = f'./{FOLDER}/{SOURCE}/val.txt'
    Lambda_weight = 0.01
    Lambda_adv = 0.001
    Lambda_local = 40
    Epsilon = 0.4
    NUM_VAL_SAMPLE = 100

if TARGET:
    INPUT_SIZE_TARGET = '512,512'
    DATA_DIRECTORY_TARGET = f'./{FOLDER}/{TARGET}'
    DATA_LIST_PATH_TARGET = f'./{FOLDER}/{TARGET}/val.txt'
    DATA_VAL_LIST_PATH_TARGET = f'./{FOLDER}/{TARGET}/val.txt'
    NUM_VAL_SAMPLE_TARGET = 100

num_classes, name_classes, mapping = readInfo(DATA_DIRECTORY)

SNAPSHOT_DIR =  f'./snapshots/{MODEL}_{SOURCE}_{ADAPTER}_2_{TARGET}_clan'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : ResNet")
    parser.add_argument("--source", type=str, default=SOURCE,
                        help="available options : GTA5, SYNTHIA")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--adapter", type=str, default=ADAPTER,
                        help="available options : ChicagoSynthetic")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size-source", type=str, default=INPUT_SIZE_SOURCE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d(NUM_CLASSES).cuda(gpu)
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)

def adjust_learning_rate(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate_D, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def weightmap(pred1, pred2):
    output = 1.0 - torch.sum((pred1 * pred2), 1).view(BATCH_SIZE, 1, pred1.size(2), pred1.size(3)) / \
    (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(BATCH_SIZE, 1, pred1.size(2), pred1.size(3))
    return output

def main():
    """Create the model and start the training."""
    writer = SummaryWriter(comment=f'{args.model}_{args.source}_2_{args.target}_clan')

    h, w = map(int, args.input_size_source.split(','))
    input_size_source = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)

    cudnn.enabled = True

    # Create Network
    if args.model == 'ResNet':
        model = Res_Deeplab(num_classes=args.num_classes)
    elif args.model == 'HRNetV2':
        model = HRNetV2(n_class=args.num_classes)
    elif args.model == 'HRNetV2OCR':
        model = HRNetV2OCR(n_class=args.num_classes, return_aug=True)

    restore_flag = False
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
        restore_flag=True
    elif args.restore_from != '':
        saved_state_dict = torch.load(args.restore_from)
        restore_flag=True

    if restore_flag:
        if args.restore_from[:4] == './mo':
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not args.num_classes == 6 or not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]

            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)

    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # Init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    # =============================================================================
    #    #for retrain
    #     saved_state_dict_D = torch.load(RESTORE_FROM_D)
    #     model_D.load_state_dict(saved_state_dict_D)
    # =============================================================================

    model_D.train()
    model_D.cuda(args.gpu)


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    shutil.copy(__file__, args.snapshot_dir)


    # Original Source Dataset
    source_dataset = CityEngineDataSet(args.data_dir, args.data_list, 
                                       max_iters=args.num_steps * args.iter_size * args.batch_size,
                                       crop_size=input_size_source, scale=False, mirror=True, mean=IMG_MEAN)
    number_item = source_dataset.item_number
    print(f"source_dataset has : {number_item}")
    # ADAPTER Dataset
    adapter_dataset = CityEngineDataSet(DATA_DIRECTORY_ADAPTER, DATA_LIST_PATH_ADAPTER, 
                                        max_iters=args.num_steps * args.iter_size * args.batch_size,
                                        crop_size=input_size_source, scale=False, mirror=True, mean=IMG_MEAN)
    number_item = adapter_dataset.item_number
    print(f"adapter_dataset has : {number_item}")
    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([source_dataset, adapter_dataset])

    
    # Create a single dataloader for the combined dataset
    trainloader = data.DataLoader(combined_dataset, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)
    
    # Assuming combined_dataset is an instance of torch.utils.data.ConcatDataset
    total_items = sum(getattr(dataset, 'item_number', 0) for dataset in combined_dataset.datasets)
    print(f"Total number of items in combined_dataset: {total_items}")


    valloader = data.DataLoader(
        CityEngineDataSet(args.data_dir, DATA_VAL_LIST_PATH,
                          max_iters=args.num_steps * args.iter_size * args.batch_size,
                          crop_size=input_size_source,
                          scale=False, mirror=True, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)
    valloader_iter = enumerate(valloader)

    targetloader = data.DataLoader(TargetDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=True, mean=IMG_MEAN),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)
    targetvalloader = data.DataLoader(
        CityEngineDataSet(args.data_dir_target, DATA_VAL_LIST_PATH_TARGET,
                          max_iters=args.num_steps * args.iter_size * args.batch_size,
                          crop_size=input_size_target,
                          scale=False, mirror=True, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    targetloader_iter = enumerate(targetloader)
    targetvalloader_iter = enumerate(targetvalloader)

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    weighted_bce_loss = WeightedBCEWithLogitsLoss()

    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # Labels for Adversarial Training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        damping = (1 - i_iter/NUM_STEPS)

        #======================================================================================
        # train G
        #======================================================================================

        #Remove Grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # Train with Source
        _, batch = next(trainloader_iter)
        images_s, labels_s, _, _, _ = batch
        images_s = Variable(images_s).cuda(args.gpu)
        if args.model == 'HRNetV2OCR':
            pred_source1, pred_source2, pred_source_aux = model(images_s)
            pred_source1 = interp_source(pred_source1)
            pred_source2 = interp_source(pred_source2)
            pred_source_aux = interp_source(pred_source_aux)
            # Segmentation Loss
            loss_seg = (loss_calc(pred_source1, labels_s, args.gpu) + loss_calc(pred_source2, labels_s, args.gpu)
                        +loss_calc(pred_source_aux, labels_s, args.gpu))

        else:
            pred_source1, pred_source2 = model(images_s)
            pred_source1 = interp_source(pred_source1)
            pred_source2 = interp_source(pred_source2)

            #Segmentation Loss
            loss_seg = (loss_calc(pred_source1, labels_s, args.gpu) + loss_calc(pred_source2, labels_s, args.gpu))
        loss_seg.backward()

        # Train with Target
        _, batch = next(targetloader_iter)
        images_t, _, _, _ = batch
        images_t = Variable(images_t).cuda(args.gpu)

        if args.model == 'HRNetV2OCR':
            pred_target1, pred_target2, pred_target_aux = model(images_t)
            pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)
            pred_target_aux = interp_source(pred_target_aux)
        else:
            pred_target1, pred_target2 = model(images_t)
            pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)

        weight_map = weightmap(F.softmax(pred_target1, dim = 1), F.softmax(pred_target2, dim = 1))

        D_out = interp_target(model_D(F.softmax(pred_target1 + pred_target2, dim = 1)))

        #Adaptive Adversarial Loss
        if(i_iter > PREHEAT_STEPS):
            loss_adv = weighted_bce_loss(D_out,
                                    Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(
                                        args.gpu), weight_map, Epsilon, Lambda_local)
        else:
            loss_adv = bce_loss(D_out,
                          Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(args.gpu))

        loss_adv = loss_adv * Lambda_adv * damping
        loss_adv.backward()

        #Weight Discrepancy Loss
        W5 = None
        W6 = None
        if args.model == 'ResNet':
            for (w5, w6) in zip(model.layer5.parameters(), model.layer6.parameters()):
                if W5 is None and W6 is None:
                    W5 = w5.view(-1)
                    W6 = w6.view(-1)
                else:
                    W5 = torch.cat((W5, w5.view(-1)), 0)
                    W6 = torch.cat((W6, w6.view(-1)), 0)
        elif args.model == 'HRNetV2' or args.model == 'HRNetV2OCR':
            for (w5, w6) in zip(model.cls1.parameters(), model.cls2.parameters()):
                if W5 is None and W6 is None:
                    W5 = w5.view(-1)
                    W6 = w6.view(-1)
                else:
                    W5 = torch.cat((W5, w5.view(-1)), 0)
                    W6 = torch.cat((W6, w6.view(-1)), 0)

        loss_weight = (torch.matmul(W5, W6) / (torch.norm(W5) * torch.norm(W6)) + 1) # +1 is for a positive loss
        loss_weight = loss_weight * Lambda_weight * damping * 2
        loss_weight.backward()

        #======================================================================================
        # train D
        #======================================================================================

        # Bring back Grads in D
        for param in model_D.parameters():
            param.requires_grad = True

        # Train with Source
        pred_source1 = pred_source1.detach()
        pred_source2 = pred_source2.detach()

        D_out_s = interp_source(model_D(F.softmax(pred_source1 + pred_source2, dim = 1)))

        loss_D_s = bce_loss(D_out_s,
                          Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda(args.gpu))

        loss_D_s.backward()

        # Train with Target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()
        weight_map = weight_map.detach()

        D_out_t = interp_target(model_D(F.softmax(pred_target1 + pred_target2, dim = 1)))

        #Adaptive Adversarial Loss
        if(i_iter > PREHEAT_STEPS):
            loss_D_t = weighted_bce_loss(D_out_t,
                                    Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(
                                        args.gpu), weight_map, Epsilon, Lambda_local)
        else:
            loss_D_t = bce_loss(D_out_t,
                          Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(args.gpu))

        loss_D_t.backward()

        optimizer.step()
        optimizer_D.step()

        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_adv = {3:.4f}, loss_weight = {4:.4f}, loss_D_s = {5:.4f} loss_D_t = {6:.4f}'.format(
            i_iter, args.num_steps, loss_seg, loss_adv, loss_weight, loss_D_s, loss_D_t))

        f_loss = open(osp.join(args.snapshot_dir,'loss_' + EXPERIENCE_NAME + '.txt'), 'a')
        f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n'.format(
            loss_seg, loss_adv, loss_weight, loss_D_s, loss_D_t))
        f_loss.close()

        scalar_info = {
            'loss_seg': loss_seg,
            'loss_adv': loss_adv,
            'loss_weight': loss_weight,
            'loss_D_s': loss_D_s,
            'loss_D_t': loss_D_t,
            'lr/lr_D1': optimizer.param_groups[0]['lr'],
            'lr/lr_D2': optimizer_D.param_groups[0]['lr']
        }

        if writer and i_iter % 10 == 0:
            for key, val in scalar_info.items():
                writer.add_scalar(key, val, i_iter)

        if writer and i_iter % 100 == 0:
            writer.add_images('source_domain/input', images_s / 256 + 0.5, i_iter)
            writer.add_images('source_domain/label', labels_s.unsqueeze(1).type(torch.float32) / NUM_CLASSES, i_iter)
            writer.add_images('source_domain/pred1',
                              torch.argmax(pred_source1, axis=1).unsqueeze(1).type(torch.float32) / NUM_CLASSES,
                              i_iter)
            writer.add_images('source_domain/pred2',
                              torch.argmax(pred_source2, axis=1).unsqueeze(1).type(torch.float32) / NUM_CLASSES,
                              i_iter)

            writer.add_images('target_domain/input', images_t / 256 + 0.5, i_iter)
            writer.add_images('target_domain/pred1',
                              torch.argmax(pred_target1, axis=1).unsqueeze(1).type(torch.float32) / NUM_CLASSES,
                              i_iter)
            writer.add_images('target_domain/pred2',
                              torch.argmax(pred_target2, axis=1).unsqueeze(1).type(torch.float32) / NUM_CLASSES,
                              i_iter)

            if args.model == 'HRNetV2OCR':
                writer.add_images('source_domain/aux',
                                  torch.argmax(pred_source_aux, axis=1).unsqueeze(1).type(torch.float32) / NUM_CLASSES,
                                  i_iter)
                writer.add_images('target_domain/aux',
                                  torch.argmax(pred_target_aux, axis=1).unsqueeze(1).type(torch.float32) / NUM_CLASSES,
                                  i_iter)


        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '_' + EXPERIENCE_NAME + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '_D_' + EXPERIENCE_NAME + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_' + EXPERIENCE_NAME + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D_' + EXPERIENCE_NAME + '.pth'))

        if i_iter % (args.save_pred_every//5) == 0 and i_iter != 0:
            with torch.no_grad():
                model.eval()
                src_mIoUs, tgt_mIoUs = [0]*num_classes, [0]*num_classes
                if valloader:
                    ii=0
                    hist = np.zeros((num_classes, num_classes))

                    for img, label, _, _, _ in valloader:
                        img = Variable(img).cuda(args.gpu)
                        if args.model == 'HRNetV2OCR':
                            pred1, pred2, _ = model(img)
                        else:
                            pred1, pred2 = model(img)
                        pred=interp_source(pred1+pred2)
                        pred = torch.argmax(pred, axis=1).detach().cpu().numpy()
                        label = label.numpy()
                        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

                        ii += 1
                        if NUM_VAL_SAMPLE>0 and ii>=NUM_VAL_SAMPLE:
                            break

                    src_mIoUs = per_class_iu(hist)

                    f_hist = open(osp.join(args.snapshot_dir, 'source_cofumat_' + EXPERIENCE_NAME + '.txt'), 'a')
                    f_hist.write(str(hist) + '\n')
                    f_hist.close()

                if targetvalloader:
                    ii = 0
                    hist = np.zeros((num_classes, num_classes))
                    for img, label, _, _, _ in targetvalloader:
                        img = Variable(img).cuda(args.gpu)
                        if args.model == 'HRNetV2OCR':
                            pred1, pred2,_ = model(img)
                        else:
                            pred1, pred2 = model(img)
                        pred = interp_source(pred1+pred2)
                        pred = torch.argmax(pred, axis=1).detach().cpu().numpy()
                        label = label.numpy()
                        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

                        ii += 1
                        if NUM_VAL_SAMPLE_TARGET>0 and ii>=NUM_VAL_SAMPLE_TARGET:
                            break
                    tgt_mIoUs = per_class_iu(hist)

                    f_hist = open(osp.join(args.snapshot_dir, 'target_cofumat_' + EXPERIENCE_NAME + '.txt'), 'a')
                    f_hist.write(str(hist) + '\n')
                    f_hist.close()
                model.train()
                print('---- Source Domain ----')
                for ind_class in range(num_classes):
                    print('===>' + name_classes[ind_class] + ':\t' + str(round(src_mIoUs[ind_class] * 100, 2)))
                print('===> mIoU: ' + str(round(np.nanmean(src_mIoUs) * 100, 2)))

                print('---- Target Domain ----')
                for ind_class in range(num_classes):
                    print('===>' + name_classes[ind_class] + ':\t' + str(round(tgt_mIoUs[ind_class] * 100, 2)))
                print('===> mIoU: ' + str(round(np.nanmean(tgt_mIoUs) * 100, 2)))

                if writer:
                    for ind_class in range(num_classes):
                        writer.add_scalar('src_iou/'+name_classes[ind_class], src_mIoUs[ind_class], i_iter)
                        writer.add_scalar('target_iou/' + name_classes[ind_class], tgt_mIoUs[ind_class], i_iter)
                    writer.add_scalar('src_iou/miou', np.nanmean(src_mIoUs),i_iter)
                    writer.add_scalar('target_iou/miou', np.nanmean(tgt_mIoUs),i_iter)

    writer.close()

if __name__ == '__main__':
    main()
