
import sys
import os
import torch
import torchvision
import numpy as np
from sklearn.utils import shuffle
import torch.nn.functional as F
import skimage.io
import skimage.transform

import config

sys.path.insert(0, "../..")


from classifier_models import PreActResNet18, ResNet18, PreActResNet34
from dataloader import get_dataloader
from networks.models import Generator, NetC_MNIST
from utils import progress_bar


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
    def __init__(self, model,opt):
        self.opt = opt
        self.model = model
        self.model.eval()
        self.build_hook()

    def build_hook(self):
        for name, m in self.model.named_children():
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

    intermedia_feature = LayerActivations(netC.to(opt.device),opt)

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
        ori_label.append(torch.argmax(preds_clean, 1).to(torch.device('cpu')))
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
    if os.path.exists(ckpt_path):
        data = torch.load(ckpt_path,map_location=opt.device)
    else:
        data = eval(netC, netG, netM, test_dl1, test_dl2, opt)

    return data


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


def gaussian_kernel(x1, x2, kernel_mul=2.0, kernel_num=5, fix_sigma=0, mean_sigma=0):
    x1_sample_size = x1.shape[0]
    x2_sample_size = x2.shape[0]
    x1_tile_shape = []
    x2_tile_shape = []
    norm_shape = []
    for i in range(len(x1.shape) + 1):
        if i == 1:
            x1_tile_shape.append(x2_sample_size)
        else:
            x1_tile_shape.append(1)
        if i == 0:
            x2_tile_shape.append(x1_sample_size)
        else:
            x2_tile_shape.append(1)
        if not (i == 0 or i == 1):
            norm_shape.append(i)

    tile_x1 = torch.unsqueeze(x1, 1).repeat(x1_tile_shape)
    tile_x2 = torch.unsqueeze(x2, 0).repeat(x2_tile_shape)
    L2_distance = torch.square(tile_x1 - tile_x2).sum(dim=norm_shape)
    # print(L2_distance)
    # bandwidth inference
    if fix_sigma:
        bandwidth = fix_sigma
    elif mean_sigma:
        bandwidth = torch.mean(L2_distance)
    else:  ## median distance
        bandwidth = torch.median(L2_distance.reshape(L2_distance.shape[0],-1))
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    print(bandwidth_list)
    #print(torch.cat(bandwidth_list,0).to(torch.device('cpu')).numpy())
    ## gaussian_RBF = exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  #L2_distance #


def kmmd_dist(x1, x2):
    X_total = torch.cat([x1,x2],0)
    Gram_matrix = gaussian_kernel(X_total,X_total,kernel_mul=2.0, kernel_num=2, fix_sigma=0, mean_sigma=0)
    n = int(x1.shape[0])
    m = int(x2.shape[0])
    x1x1 = Gram_matrix[:n, :n]
    x2x2 = Gram_matrix[n:, n:]
    x1x2 = Gram_matrix[:n, n:]
    # x2x1 = Gram_matrix[n:, :n]  # Gram_matrix is symmetric
    diff = torch.mean(x1x1) + torch.mean(x2x2) - 2 * torch.mean(x1x2)
    diff = (m*n)/(m+n)*diff
    return diff.to(torch.device('cpu')).numpy()

class Feature_Correlations:
    def __init__(self,POWER_list, mode='mad'):
        self.power = POWER_list
        self.mode = mode

    def train(self, in_data):
        self.in_data = in_data
        if 'mad' in self.mode:
            self.medians, self.mads = self.get_median_mad(self.in_data)
            self.mins, self.maxs = self.minmax_mad()


    def minmax_mad(self):
        mins = []
        maxs = []
        for L, mm in enumerate(zip(self.medians,self.mads)):
            medians, mads = mm[0], mm[1]
            if L==len(mins):
                mins.append([None]*len(self.power))
                maxs.append([None]*len(self.power))
            for p, P in enumerate(self.power):
                    mins[L][p] = medians[p]-mads[p]*10
                    maxs[L][p] = medians[p]+mads[p]*10
        return mins, maxs

    def G_p(self, ob, p):
        temp = ob.detach()
        temp = temp**p
        temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
        temp = temp.triu()
        temp = temp.sign()*torch.abs(temp)**(1/p)
        temp = temp.reshape(temp.shape[0],-1)
        self.num_feature = temp.shape[-1]/2
        return temp

    def get_median_mad(self, feat_list):
        medians = []
        mads = []
        for L,feat_L in enumerate(feat_list):
            if L==len(medians):
                medians.append([None]*len(self.power))
                mads.append([None]*len(self.power))
            for p,P in enumerate(self.power):
                g_p = self.G_p(feat_L,P)
                current_median = g_p.median(dim=0,keepdim=True)[0]
                current_mad = torch.abs(g_p - current_median).median(dim=0,keepdim=True)[0]
                medians[L][p] = current_median
                mads[L][p] = current_mad
        return medians, mads

    def get_deviations_(self, feat_list):
        deviations = []
        batch_deviations = []
        for L,feat_L in enumerate(feat_list):
            dev = 0
            for p,P in enumerate(self.power):
                g_p = self.G_p(feat_L,P)
                dev +=  (F.relu(self.mins[L][p]-g_p)/torch.abs(self.mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
                dev +=  (F.relu(g_p-self.maxs[L][p])/torch.abs(self.maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
            batch_deviations.append(dev.cpu().detach().numpy())
        batch_deviations = np.concatenate(batch_deviations,axis=1)
        deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0) /self.num_feature /len(self.power)
        return deviations

    def get_deviations(self, feat_list):
        deviations = []
        batch_deviations = []
        for L,feat_L in enumerate(feat_list):
            dev = 0
            for p,P in enumerate(self.power):
                g_p = self.G_p(feat_L,P)
                dev += torch.sum(torch.abs(g_p-self.medians[L][p])/(self.mads[L][p]+1e-6),dim=1,keepdim=True)
            batch_deviations.append(dev.cpu().detach().numpy())
        batch_deviations = np.concatenate(batch_deviations,axis=1)
        deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0)/self.num_feature /len(self.power)
        return deviations


def threshold_determine(clean_feature_target, ood_detection):
    test_deviations_list = []
    step = 5
    for i in range(step):
        index_mask = np.ones((len(clean_feature_target),))
        index_mask[i*int(len(clean_feature_target)//step):(i+1)*int(len(clean_feature_target)//step)] = 0
        clean_feature_target_train= clean_feature_target[np.where(index_mask == 1)]
        clean_feature_target_test = clean_feature_target[np.where(index_mask == 0)]
        ood_detection.train(in_data=[clean_feature_target_train],)
        test_deviations = ood_detection.get_deviations_([clean_feature_target_test])
        test_deviations_list.append(test_deviations)
    test_deviations = np.concatenate(test_deviations_list,0)
    test_deviations_sort = np.sort(test_deviations,0)
    percentile_95 = test_deviations_sort[int(len(test_deviations_sort)*0.95)][0]
    percentile_99 = test_deviations_sort[int(len(test_deviations_sort)*0.99)][0]
    print(f'percentile_95:{percentile_95}')
    print(f'percentile_99:{percentile_99}')
    # plt.hist(test_deviations)
    # plt.show()
    return percentile_95, percentile_99

class BEAT_detector():
    def __init__(self, opt, clean_test=500,bd_test=500,order_list=np.arange(1,9)):
        self.opt = opt
        self.test_target_label = opt.target_label
        self.order_list = order_list
        if opt.dataset == 'cifar10':
            self.clean_data_perclass = 30
        elif opt.dataset == 'gtsrb':
            self.clean_data_perclass = 30
        else:
            raise Exception("Invalid dataset")

        self.clean_test = clean_test
        self.bd_test = bd_test


    def _detecting(self,data):
        opt = self.opt
        clean_feature = data['clean_feature'].to(opt.device)
        bd_feature = data['bd_feature'].to(opt.device)
        ori_label = data['ori_label'].cpu()
        bd_label = data['bd_label'].cpu()

        (clean_feature,bd_feature,ori_label,bd_label) = shuffle(clean_feature,bd_feature,ori_label,bd_label)

        ##### use gram-matrix OOD detection

        ood_detection = Feature_Correlations(POWER_list=self.order_list,mode='mad')

        J_t = []
        threshold_list = []

        for test_target_label in range(opt.num_classes):
            print(f'*****class:{test_target_label}*****')
            clean_feature_target = clean_feature[np.where(ori_label==test_target_label)]
            clean_feature_defend = clean_feature_target[:self.clean_data_perclass]

            threshold_95, threshold_99 = threshold_determine(clean_feature_defend, ood_detection)
            threshold_list.append([test_target_label,threshold_95, threshold_99])

            ood_detection.train(in_data=[clean_feature_defend])
            clean_feature_test = clean_feature[np.where(ori_label==test_target_label)][-self.clean_test:]
            clean_label_test = np.zeros((clean_feature_test.shape[0],))

            if test_target_label == opt.target_label:
                bd_feature_test,  _ = bd_dataset(bd_feature,ori_label,n_poison=self.bd_test,num_class=opt.num_classes,target_class=[test_target_label],source_class=list(np.arange(opt.num_classes)),balance=False)
                bd_label_test = np.ones((bd_feature_test.shape[0],))
                feature_test = torch.cat([clean_feature_test,bd_feature_test],0)
                label_test = np.concatenate([clean_label_test,bd_label_test],0)

                clean_deviations_sort = np.sort(ood_detection.get_deviations_([clean_feature_test]),0)
                bd_deviations_sort   = np.sort(ood_detection.get_deviations_([bd_feature_test]),0)
                percentile_95 = np.where(bd_deviations_sort > clean_deviations_sort[int(len(clean_deviations_sort)*0.95)], 1, 0)
                print(f'percentile_95:{clean_deviations_sort[int(len(clean_deviations_sort)*0.95)]},TP95:{percentile_95.sum()/len(bd_deviations_sort)}')
                percentile_99 = np.where(bd_deviations_sort > clean_deviations_sort[int(len(clean_deviations_sort)*0.99)], 1, 0)
                print(f'percentile_99:{clean_deviations_sort[int(len(clean_deviations_sort)*0.99)]},TP99:{percentile_99.sum()/len(bd_deviations_sort)}')
            else:
                feature_test = clean_feature_test
                label_test = clean_label_test


            test_deviations = ood_detection.get_deviations_([feature_test])

            ood_label_95 = np.where(test_deviations > threshold_95, 1, 0).squeeze()
            ood_label_99 = np.where(test_deviations > threshold_99, 1, 0).squeeze()

            false_negetive_95 = np.where(label_test - ood_label_95 > 0, 1, 0).squeeze()
            false_negetive_99 = np.where(label_test - ood_label_99 > 0, 1, 0).squeeze()
            false_positive_95 = np.where(label_test - ood_label_95 < 0, 1, 0).squeeze()
            false_positive_99 = np.where(label_test - ood_label_99 < 0, 1, 0).squeeze()

            print(f'false_negetive_95:{false_negetive_95.sum()},false_negetive_99:{false_negetive_99.sum()}')
            print(f'false_positive_95:{false_positive_95.sum()},false_positive_99：{false_positive_99.sum()}')


            clean_feature_group = feature_test[np.where(ood_label_95==0)]
            bd_feature_group = feature_test[np.where(ood_label_95==1)]

            clean_feature_flat = torch.mean(clean_feature_group,dim=(2,3))
            bd_feature_flat = torch.mean(bd_feature_group,dim=(2,3))
            if bd_feature_flat.shape[0] < 1:
                kmmd = np.array([0.0])
            else:
                kmmd = kmmd_dist(clean_feature_flat, bd_feature_flat)
            print(f'KMMD:{kmmd.item()}.')

            J_t.append(kmmd.item())

        print(J_t)
        J_t = np.asarray(J_t)
        J_t_median = np.median(J_t)
        J_MAD = np.median(np.abs(J_t - J_t_median))
        J_star = np.abs(J_t - J_t_median)/1.4826/(J_MAD+1e-6)
        # [print('class %d: %.2f'%(i,J_star_i)) for i,J_star_i in enumerate(J_star)]
        [print('%.2f'%(J_star_i)) for i,J_star_i in enumerate(J_star)]
        self._save_result_to_dir(result=[J_star])


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


def main(k):
    opt = config.get_argument().parse_args()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(opt.device)
    opt.target_label = k
    print('-'*50+'opt.target_label:',opt.target_label)
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
    beat_detector = BEAT_detector(opt)
    beat_detector._detecting(data)

if __name__ == "__main__":
    opt = config.get_argument().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    for k in range(10):
        main(k)

## python Beatrix.py --dataset cifar10 --gpu 0
