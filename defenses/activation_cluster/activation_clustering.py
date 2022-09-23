# -*- coding: utf-8 -*-
"""
Created on 16:58, 10-09-2021

@author: Wanlun Ma
"""

import sys
import torch
import torchvision
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, FastICA
from sklearn.cluster import kmeans_plusplus, KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import shuffle
import numpy as np
import skimage.io
import skimage.transform
import os


import config

sys.path.insert(0, "../..")


from classifier_models import PreActResNet18, ResNet18, PreActResNet34
from dataloader import get_dataloader
from networks.models import Generator, NetC_MNIST
from utils import progress_bar

def bd_dataset(X_all,y_all,n_poison=100,num_class=10,target_class=[],source_class=[1],balance=True):
    #X_all,y_all = sklearn.utils.shuffle(X_all,y_all)
    if len(y_all.shape) >1:
        print('y_all.shape:',y_all.shape)
        labels = np.argmax(y_all, axis=-1)
    else:
        labels = y_all
    train_classes = np.arange(num_class)
    clean_x = []
    clean_y = []
    poison_y = []
    if balance:
        for tc in train_classes: ## poison mode
            if tc in target_class:
                continue
            if tc not in source_class:
                continue
            index = np.where(labels == tc)
            clean_x.append(X_all[index][0:n_poison])
            clean_y.append(y_all[index][0:n_poison])
            poison_index = np.where(labels == target_class[0])
            poison_y.append(y_all[poison_index][0:n_poison])
        clean_x = torch.cat(clean_x, dim=0)
        clean_y = torch.cat(clean_y, dim=0)
        poison_y = torch.cat(poison_y, dim=0)
    else:
        index = np.where(labels != target_class[0])
        clean_x.append(X_all[index][0:n_poison])
        clean_y.append(y_all[index][0:n_poison])
        poison_index = np.where(labels == target_class[0])
        poison_y.append(y_all[poison_index][0:n_poison])
        clean_x = torch.cat(clean_x, dim=0)
        clean_y = torch.cat(clean_y, dim=0)
        poison_y = torch.cat(poison_y, dim=0)

    return clean_x, poison_y

class Activation_Clustering():
    def __init__(self, opt):
        self.transformer = FastICA(n_components=10,max_iter=1000)
        self.opt = opt
    def run_detection(self, data):
        clean_feature = data['clean_feature']
        bd_feature = data['bd_feature']
        # cross_feature = data['cross_feature']
        ori_label = data['ori_label'].cpu()
        bd_label = data['bd_label'].cpu()
        (clean_feature,bd_feature,ori_label,bd_label) = shuffle(clean_feature,bd_feature,ori_label,bd_label)
        for test_target_label in range(self.opt.num_classes):
            clean_feature_test = clean_feature[np.where(ori_label==test_target_label)][:200]
            clean_label_test = np.zeros((clean_feature_test.shape[0],))
            if bd_label[0] == test_target_label:
                bd_feature_test,  _ = bd_dataset(bd_feature,ori_label,n_poison=200,num_class=self.opt.num_classes,target_class=[test_target_label],source_class=list(np.arange(self.opt.num_classes)),balance=False)
                bd_label_test = np.ones((bd_feature_test.shape[0],))
                feature_test = torch.cat([clean_feature_test,bd_feature_test],0)
                label_test = np.concatenate([clean_label_test,bd_label_test],0)
            else:
                feature_test = clean_feature_test
                label_test = clean_label_test

            feature_test = torch.mean(feature_test,dim=(2,3))
            X = feature_test.cpu().numpy()
            transformed_X = self.transformer.fit_transform(X)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(transformed_X)
            silhouette = silhouette_score(transformed_X,kmeans.labels_)
            print(f"class:{test_target_label}: silhouette score:{silhouette}")
            result = [silhouette, test_target_label, self.opt.target_label]
            save_result_to_dir(self.opt,result)

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


    data = {"clean_feature": torch.cat(clean_feature,dim=0),
            "bd_feature": torch.cat(bd_feature,0),
            "ori_label": torch.cat(ori_label,0),
            "bd_label": torch.cat(bd_label,0),
            }
    # ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, 'target_'+str(opt.target_label))
    # ckpt_path = os.path.join(ckpt_folder, 'data.pt')
    # torch.save(data,ckpt_path)
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

    data = eval(netC, netG, netM, test_dl1, test_dl2, opt)
    return data

def detection(opt):
    data = train(opt)
    AC_detector = Activation_Clustering(opt)
    AC_detector.run_detection(data)

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

    detection(opt)

if __name__ == "__main__":
    opt = config.get_argument().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    for k in range(0,10):
        main(k)
## python activation_clustering.py --dataset cifar10 --gpu 0
