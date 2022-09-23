# -*- coding: utf-8 -*-
"""
Created on 16:58, 10-09-2021

@author: Wanlun Ma
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import torch
import torchvision
import numpy as np
from sklearn.utils import shuffle
import skimage.io
import skimage.transform

from statistical_contamination_analyzer import clean_dataset, bd_dataset, two_decom_cov, two_decom, two_sub
import config


sys.path.insert(0, "../..")


from classifier_models import PreActResNet18, ResNet18, PreActResNet34
from dataloader import get_dataloader
from networks.models import Generator, NetC_MNIST
from utils import progress_bar


def save_result_to_dir(opt, result):
    result_dir = os.path.join(opt.result, opt.dataset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, opt.attack_mode, 'target_'+str(opt.target_label))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output_path = os.path.join(result_path, "{}_{}_output.txt".format(opt.attack_mode, opt.dataset))
    with open(output_path, "a+") as f:
        f.write(f'silhouette_score: {str(result[0])}, test_target_label: {str(result[1])},true_target_label: {str(result[2])}\n')


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all_mask":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)

def create_bd(inputs, targets, netG, netM, opt):
    bd_targets = create_targets_bd(targets, opt)
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)

    masks_output = netM.threshold(netM(inputs))
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs, bd_targets, patterns, masks_output

def create_cross(inputs1, inputs2, netG, netM, opt):
    patterns2 = netG(inputs2)
    patterns2 = netG.normalize_pattern(patterns2)
    masks_output = netM.threshold(netM(inputs2))
    inputs_cross = inputs1 + (patterns2 - inputs1) * masks_output
    return inputs_cross, patterns2, masks_output

class LayerActivations:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.build_hook()

    def build_hook(self):
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Sequential) and name == 'layer4':
                self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = input[0]

    def remove_hook(self):
        self.hook.remove()

    def run_hook(self,x):
        self.model(x)
        # self.remove_hook()
        return self.features

def eval(netC, netG, netM, test_dl1, test_dl2, opt):
    print(" Eval:")

    n_output_batches = 3
    n_output_images = 3
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0
    total_correct_cross = 0

    clean_feature = []
    bd_feature = []
    cross_feature = []
    ori_label = []
    bd_label = []

    intermedia_feature = LayerActivations(netC.to(opt.device))
    for batch_idx, (inputs, targets), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
        inputs1, targets1 = inputs.to(opt.device), targets.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
        bs = inputs1.shape[0]
        total_sample += bs

        # Evaluating clean
        preds_clean = netC(inputs1)
        correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets1)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100.0 / total_sample

        # Evaluating backdoor
        if opt.attack_mode == "all2one":
            inputs_bd, targets_bd, _, _ = create_bd(inputs1, targets1, netG, netM, opt)
            preds_bd = netC(inputs_bd)
            correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            total_correct_bd += correct_bd
            acc_bd = total_correct_bd * 100.0 / total_sample

            inputs_cross, _, _ = create_cross(inputs1, inputs2, netG, netM, opt)
            preds_cross = netC(inputs_cross)
            correct_cross = torch.sum(torch.argmax(preds_cross, 1) == targets1)
            total_correct_cross += correct_cross
            acc_cross = total_correct_cross * 100.0 / total_sample
        else:
            raise Exception("Invalid attack mode")

        clean_feature.append(intermedia_feature.run_hook(inputs1).to(torch.device('cpu')))
        bd_feature.append(intermedia_feature.run_hook(inputs_bd).to(torch.device('cpu')))
        # cross_feature.append(netC.intermedia_feature(inputs_cross).to(torch.device('cpu')))
        ori_label.append(targets1.to(torch.device('cpu')))
        bd_label.append(torch.argmax(preds_bd, 1).to(torch.device('cpu')))

        progress_bar(
            batch_idx,
            len(test_dl1),
            "Acc Clean: {:.3f} | Acc Bd: {:.3f} | Acc Cross: {:.3f}".format(acc_clean, acc_bd, acc_cross),
        )

        if batch_idx < n_output_batches:
            dir_temps = os.path.join(opt.temps, opt.dataset)
            if not os.path.exists(dir_temps):
                os.makedirs(dir_temps)
            subs = []
            for i in range(n_output_images):
                subs.append(inputs_bd[i : (i + 1), :, :, :])
            images = netG.denormalize_pattern(torch.cat(subs, dim=3))
            file_name = "%s_%s_sample_%d.png" % (opt.dataset, opt.attack_mode, batch_idx)
            file_path = os.path.join(dir_temps, file_name)
            torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)


    data = {"clean_feature": torch.mean(torch.cat(clean_feature,dim=0),dim=(2,3)),
            "bd_feature": torch.mean(torch.cat(bd_feature,0),dim=(2,3)),
            # "cross_feature": torch.cat(cross_feature,0),
            "ori_label": torch.cat(ori_label,0),
            "bd_label": torch.cat(bd_label,0),
            }

    if opt.save_feature_data:
        dir_data = 'feature_data/'
        ckpt_folder = os.path.join(dir_data, opt.dataset, opt.attack_mode, 'target_'+str(opt.target_label))
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        ckpt_path = os.path.join(ckpt_folder, 'data.pt')
        torch.save(data,ckpt_path)
    return data


def train(opt):
    # Prepare model related things
    if opt.dataset == "cifar10":
        netC = PreActResNet18().to(opt.device)
    elif opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=43).to(opt.device)
    elif opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
    else:
        raise Exception("Invalid dataset")

    # For tensorboard
    log_dir = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, 'target_'+str(opt.target_label))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, "log_dir")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # tf_writer = SummaryWriter(log_dir=log_dir)

    # Continue training ?
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, 'target_'+str(opt.target_label))
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))

    state_dict = torch.load(ckpt_path,map_location=opt.device)
    print("load C")
    netC.load_state_dict(state_dict["netC"])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    print("load G")
    netG = Generator(opt)
    netG.load_state_dict(state_dict["netG"])
    netG.to(opt.device)
    netG.eval()
    netG.requires_grad_(False)
    print("load M")
    netM = Generator(opt, out_channels=1)
    netM.load_state_dict(state_dict["netM"])
    netM.to(opt.device)
    netM.eval()
    netM.requires_grad_(False)

    opt.n_iters = 1
    opt.batchsize = 256
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    test_dl1 = get_dataloader(opt, train=False)
    test_dl2 = get_dataloader(opt, train=False)

    dir_data = 'feature_data/'
    ckpt_folder = os.path.join(dir_data, opt.dataset, opt.attack_mode, 'target_'+str(opt.target_label))
    ckpt_path = os.path.join(ckpt_folder, 'data.pt')
    print(ckpt_path)
    if os.path.exists(ckpt_path):
        data = torch.load(ckpt_path,map_location=opt.device)
    else:
        print('generate feature data...')
        data = eval(netC, netG, netM, test_dl1, test_dl2, opt)

    return data



class SCAN_detector():
    def __init__(self, opt,clean_test=200,bd_test=150):
        self.opt = opt
        self.test_target_label = opt.target_label
        if opt.dataset == "cifar10":
            self.clean_data_perclass = 1000//opt.num_classes
        elif opt.dataset == "gtsrb":
            self.clean_data_perclass = 1000//opt.num_classes
        else:
            raise Exception("Invalid Dataset")

        self.clean_test = clean_test
        self.bd_test = bd_test


    def _detecting(self,data):
        opt = self.opt
        clean_feature = data['clean_feature'].to(opt.device)
        bd_feature = data['bd_feature'].to(opt.device)
        # cross_feature = data['cross_feature']
        ori_label = data['ori_label'].cpu()
        bd_label = data['bd_label'].cpu()

        if len(clean_feature.shape) == 4:
            clean_feature = torch.mean(clean_feature,dim=(2,3))
            bd_feature = torch.mean(bd_feature,dim=(2,3))
        elif len(clean_feature.shape) == 2:
            clean_feature = clean_feature
            bd_feature = bd_feature
        else:
            raise Exception("Invalid data shape!")

        laten_dim = clean_feature.shape[1]

        (clean_feature,bd_feature,ori_label,bd_label) = shuffle(clean_feature,bd_feature,ori_label,bd_label)

        clean_x, clean_y, clean_feature_test, test_label_clean = clean_dataset(clean_feature, ori_label,
                                                                               n_clean=self.clean_data_perclass,
                                                                               n_clean_test=self.clean_test,
                                                                               num_class=opt.num_classes)

        clean_x = clean_x - np.mean(clean_x,axis=0,keepdims=True)

        u_start, e_start, S_u, S_e = two_decom_cov(clean_x, clean_y,max_iter=200,balance_sample=False)

        bd_feature_test, test_label_bd_ori, test_label_bd = bd_dataset(bd_feature,ori_label,bd_label,
                                                                       n_poison=self.bd_test,num_class=opt.num_classes,
                                                                       target_class=[self.test_target_label],
                                                                       source_class=list(np.arange(opt.num_classes)),
                                                                       balance=False)

        feature_test = np.concatenate([clean_feature_test,bd_feature_test],0)
        feature_test = feature_test - np.mean(feature_test,axis=0,keepdims=True)
        label_test = np.concatenate([test_label_clean,test_label_bd],0)
        label_test_ori = np.concatenate([test_label_clean,test_label_bd_ori],0)

        u0, e0 = two_decom(feature_test, label_test, S_u, S_e)

        J_t = []
        y_t_ori_all = []
        r_projected_all = []
        y_sub_all = []
        class_num, labels = tf.unique(label_test)
        F = tf.linalg.pinv(S_e)
        for i in range(len(class_num)):
            # if i > 2:
            #     break
            # print(f'*****class:{i}*****')
            index = tf.where(labels == tf.cast(class_num[i], dtype=tf.int32))
            # print(f'index:{index.shape}')
            r_t = tf.gather_nd(feature_test, indices=index)
            #r_t = r_t - tf.reduce_mean(r_t, axis=0, keepdims=True) ## centralize in class
            y_t = tf.gather_nd(label_test, indices=index)
            u_sub, y_sub, r_projected = two_sub(r_t,y_t,S_e,S_u,max_iter=200)
            if i == self.test_target_label:
                y_t_ori = tf.gather_nd(label_test_ori, indices=index).numpy()
                r_projected = np.squeeze(r_projected)
                y_t_ori_all.append(y_t_ori)
                r_projected_all.append(r_projected)
                y_sub_all.append(y_sub)
            J1 = tf.linalg.matmul(tf.linalg.matmul((r_t - u0[i]),F),(r_t - u0[i]),transpose_b=True)
            J2 = tf.linalg.matmul(tf.linalg.matmul((r_t - u_sub),F),(r_t - u_sub),transpose_b=True)
            J = tf.linalg.trace(J1 - J2)
            J_t.append((J.numpy()-laten_dim) / np.sqrt(2*laten_dim))
            print(tf.linalg.trace(J1).numpy(),tf.linalg.trace(J2).numpy(),J.numpy())
            print(f'class:{i},J_t:{J_t}')

        y_t_ori_all = np.concatenate(y_t_ori_all,0)
        r_projected_all = np.concatenate(r_projected_all,0)
        y_sub_all = np.concatenate(y_sub_all,0)
        print('J_t:',J_t)
        J_t = np.asarray(J_t)
        J_t_median = np.median(J_t)
        J_MAD = np.median(np.abs(J_t - J_t_median))
        J_star =np.abs(J_t - J_t_median)/J_MAD/1.4826
        [print('%.2f'%(J_star_i)) for J_star_i in J_star]
        self._save_result_to_dir(result=[J_star,r_projected_all,y_t_ori_all, y_sub_all])

    def _save_result_to_dir(self, result):
        opt = self.opt
        result_dir = os.path.join(opt.result, opt.dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, opt.attack_mode, 'target_'+str(opt.target_label))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        output_path = os.path.join(result_path, "{}_{}_output.txt".format(opt.attack_mode, opt.dataset))
        with open(output_path, "w+") as f:
            J_star_to_save = [str(value) for value in result[0]]
            f.write(", ".join(J_star_to_save) + "\n")
            r_projected_to_save = [str(value) for value in result[1]]
            f.write(", ".join(r_projected_to_save) + "\n")
            y_ori_to_save = [str(value) for value in result[2]]
            f.write(", ".join(y_ori_to_save) + "\n")
            y_sub_to_save = [str(value) for value in result[3]]
            f.write(", ".join(y_sub_to_save) + "\n")

def main(k):
    opt = config.get_argument().parse_args()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(opt.device)
    opt.target_label = k
    print('-'*20+'opt.target_label:',opt.target_label)
    if opt.dataset == "mnist" or opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    else:
        raise Exception("Invalid Dataset")

    data = train(opt)
    # scan_detector = SCAN_detector(opt)
    # scan_detector._detecting(data)

if __name__ == "__main__":
    opt = config.get_argument().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    for k in range(43):
        main(k)

## python scanner.py --dataset cifar10 --gpu 1 --attack_mode badnet
## python scanner.py --dataset gtsrb --gpu 1 --attack_mode badnet
