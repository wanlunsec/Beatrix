
import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
from scipy.optimize import curve_fit
from scipy.optimize import fmin_cobyla
from tqdm import tqdm
import skimage.io
import skimage.transform
import argparse
import time

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from preact_resnet import PreActResNet18
from networks.models import Generator, NetC_MNIST


def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--data_root", type=str, default="../../data/")
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--trigger_path", type=str, default="./triggers/")

    # ---------------------------- For SentiNet --------------------------
    # Model hyperparameters
    parser.add_argument("--n_sample", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--test_rounds", type=int, default=10)

    parser.add_argument("--true_target_label", type=int,default=0)
    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')
    parser.add_argument("--sentinet_mode", default='0',type=str)
    parser.add_argument("--MASK_COND", default=0.85,type=float)

    return parser

class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = (x[:, :, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = x[:, :, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone



class GTSRB(torch.utils.data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Final_Training/Images")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Final_Test/Images")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label


def get_transform(opt, train=True):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if train:
        transforms_list.append(
            transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.input_height // 8)
        )
        transforms_list.append(transforms.RandomRotation(10))
        transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)

def get_dataset(opt, train=True):
    if opt.dataset == "gtsrb":
        dataset = GTSRB(
            opt,
            train,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform=ToNumpy(), download=True)
    else:
        raise Exception("Invalid dataset")
    return dataset


def get_dataloader(opt, train=True):
    transform = get_transform(opt, train)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.BATCH_SIZE, num_workers=opt.num_workers, shuffle=True)
    return dataloader


def read_trigger(opt):
    if opt.attack_mode == "badnet":
        p_size = 4
        loc_x = opt.input_height - p_size
        loc_y = opt.input_width - p_size
        patterns = torch.zeros(1,opt.input_channel,opt.input_height, opt.input_width).to(opt.device)
        patterns[:, :, loc_x:loc_x+p_size, loc_y:loc_y+p_size] = 1.0
    elif opt.attack_mode == "badnet_normal":
        fn = os.path.join(opt.trigger_path+'normal_md.png')
        img = skimage.io.imread(fn)
        img = skimage.transform.resize(img, (opt.input_height, opt.input_width))
        img = np.stack([img,img,img],0)
        patterns = torch.tensor(img)
        patterns = torch.unsqueeze(patterns,0)
    elif opt.attack_mode == "badnet_watermark":
        fn = os.path.join(opt.trigger_path+'trojan_watermark.jpg')
        img = skimage.io.imread(fn)
        img = skimage.transform.resize(img, (opt.input_height, opt.input_width))
        img = np.transpose(img, (2, 0, 1))
        patterns = torch.tensor(img)
        patterns = torch.unsqueeze(patterns,0)
    elif opt.attack_mode == "badnet_square":
        fn = os.path.join(opt.trigger_path+'trojan_square.jpg')
        img = skimage.io.imread(fn)
        img = skimage.transform.resize(img, (opt.input_height, opt.input_width))
        img = np.transpose(img, (2, 0, 1))
        patterns = torch.tensor(img)
        patterns = torch.unsqueeze(patterns,0)
    else:
        raise Exception("Invalid attack mode")
    patterns = patterns.to(dtype=torch.float32)
    return patterns.to(opt.device)

class SentiNet:
    def __init__(self, opt):
        super().__init__()
        self.n_sample = opt.n_sample
        self.normalizer = self._get_normalize(opt)
        self.denormalizer = self._get_denormalize(opt)
        self.device = opt.device
        self.input_height,self.input_width,self.input_channel = opt.input_height,opt.input_width,opt.input_channel

    def _get_denormalize(self, opt):
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, opt):
        if opt.dataset == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        if normalizer:
            transform = transforms.Compose([transforms.ToTensor(), normalizer])
        else:
            transform = transforms.ToTensor()
        return transform

    def normalize(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x

    def denormalize(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x

    def _superimpose(self, background, overlay, mask):
        output = background*mask + overlay*(1.0 - mask)
        output = np.clip(output, 0, 255).astype(np.uint8)
        assert len(output.shape) == 3
        return output

    def _get_entropy(self, background, mask, dataset, classifier, target_label):
        index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
        inert_pattern = np.random.rand(self.n_sample,self.input_height,self.input_width,self.input_channel) * 255.0
        inert_pattern = np.clip(inert_pattern, 0, 255).astype(np.uint8)
        x1_add = [0] * self.n_sample
        x2_add = [0] * self.n_sample
        for index in range(self.n_sample):
            add_image = self._superimpose(background, dataset[index_overlay[index]][0], mask)
            add_image = self.normalize(add_image)
            x1_add[index] = add_image
            ip_image = self._superimpose(background, inert_pattern[index], mask)
            ip_image = self.normalize(ip_image)
            x2_add[index] = ip_image
        py1_add = classifier(torch.stack(x1_add).to(self.device))
        py2_add = classifier(torch.stack(x2_add).to(self.device))
        py2_add = F.softmax(py2_add, dim=1)
        _, yR = torch.max(py1_add, 1)
        conf_ip, _ = torch.max(py2_add, 1)
        # print(f'conf_ip:{conf_ip.shape}, conf_ip:{conf_ip},yR:{yR},target_label:{target_label}')
        fooled_y = torch.sum(torch.where(yR==target_label,1.0,0.0))/self.n_sample
        avg_conf_ip = torch.mean(conf_ip)
        return fooled_y.detach().cpu().numpy(), avg_conf_ip.detach().cpu().numpy()

    def __call__(self, background, mask, dataset, classifier, target_label):
        return self._get_entropy(background, mask, dataset, classifier, target_label)


def sentinet(opt, mode='attack'):
    if opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    else:
        raise Exception("Invalid dataset")

    opt.BATCH_SIZE = opt.n_test
    # Prepare test set
    testset = get_dataset(opt, train=False)
    test_dataloader = get_dataloader(opt, train=False)

    # SentiNet detector
    sentinet_detector = SentiNet(opt)

    # Prepare pretrained classifier
    if opt.dataset == "mnist":
        netC = NetC_MNIST()
    elif opt.dataset == "cifar10":
        netC = PreActResNet18()
    elif opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=43)
    else:
        raise Exception("Invalid dataset")
    # for param in netC.parameters():
    #     param.requires_grad = False
    netC.to(opt.device)
    netC.eval()

    netG = Generator(opt)
    for param in netG.parameters():
        param.requires_grad = False
    netG.to(opt.device)
    netG.eval()

    # Load pretrained model
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, 'target_'+str(opt.true_target_label))
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    print('-'*20,ckpt_path)
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=torch.device(opt.device))
        netC.load_state_dict(state_dict["netC"])
        netG.load_state_dict(state_dict["netG"])
        netM = Generator(opt, out_channels=1)
        netM.load_state_dict(state_dict["netM"])
        netM.to(opt.device)
        netM.eval()
        netM.requires_grad_(False)
    else:
        raise Exception(f"Invalid ckpt path:{ckpt_path}")


    print("Testing with bd data !!!!")

    inputs, targets = next(iter(test_dataloader))

    inputs = inputs.to(opt.device)

    images = netG.denormalize_pattern(inputs) *255.0
    images = images.detach().cpu().numpy()
    images = np.clip(images, 0, 255).astype(np.uint8).transpose((0, 2, 3, 1))

    ### add trigger

    if opt.attack_mode == "all2one":
        patterns = netG(inputs)
        patterns = netG.normalize_pattern(patterns)
        batch_masks = netM.threshold(netM(inputs))
        bd_inputs = inputs + (patterns - inputs) * batch_masks
    elif opt.attack_mode in ["badnet", "badnet_normal","badnet_square","badnet_watermark"]:
        trigger_pattern = read_trigger(opt)
        alpha = 1.0
        patterns = trigger_pattern.clone()
        patterns = patterns.repeat(inputs.shape[0],1,1,1)
        masks_output = patterns.clone()
        patterns = netG.normalize_pattern(patterns)
        batch_masks = masks_output*alpha
        if opt.attack_mode == "badnet":
            bd_inputs = inputs + (patterns - inputs) * batch_masks
        else:
            bd_inputs = inputs + patterns
    else:
        raise Exception("Invalid attack mode")


    bd_images = netG.denormalize_pattern(bd_inputs) * 255.0
    bd_images = bd_images.detach().cpu().numpy()
    bd_images = np.clip(bd_images, 0, 255).astype(np.uint8).transpose((0, 2, 3, 1))
    print(f'bd_images.shape: {bd_images.shape}')

    bd_masks = netG.denormalize_pattern(batch_masks) * 255.0
    bd_masks = bd_masks.detach().cpu().numpy()
    bd_masks = np.clip(bd_masks, 0, 255).astype(np.uint8).transpose((0, 2, 3, 1))
    bd_masks = np.where(bd_masks>0,1,0).astype(np.uint8)


    target_layer = netC.layer4[-1]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=netC, target_layer=target_layer, use_cuda=opt.use_cuda)


    if mode == "attack":
        input_tensor = bd_inputs
        input_images = bd_images
    else:
        input_tensor = inputs
        input_images = images

    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    print(f'input_tensor.shape:{input_tensor.shape}')
    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    print(f'true label: {targets.numpy()}')
    target_category = None #None #opt.true_target_label #
    t1 = time.time()
    grayscale_cams = cam(input_tensor=input_tensor, target_category=target_category)
    masks = np.where(grayscale_cams >= opt.MASK_COND,1,0)
    masks = np.expand_dims(masks,axis=-1) # add 1D to mask
    t3 = time.time()
    print(f'runtime per sample for GradCAM: {(t3-t1)/opt.n_test}')

    if opt.use_truemask and mode=='attack': ###following the setting in SCAn(USENIX'21)
        masks = bd_masks

    # print('masks.shape:',masks.shape)

    if opt.show_gradcam:
        grayscale_cam = grayscale_cams[0, :]
        plt.imshow(grayscale_cam)
        plt.savefig('log/{}_{}_{}_dynamic_mask.png'.format(opt.dataset,opt.attack_mode,mode))

        mask = np.where(grayscale_cam >= opt.MASK_COND,1,0)
        plt.imshow(mask)
        # plt.show()
        plt.savefig('log/{}_{}_{}_dynamic_condmask.png'.format(opt.dataset,opt.attack_mode,mode))

        rgb_img = input_images[0,:] / 255.0
        plt.imshow(rgb_img)
        plt.savefig('log/{}_{}_{}_dynamic_image.png'.format(opt.dataset,opt.attack_mode,mode))
        plt.close()

        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        plt.imshow(cam_image)
        plt.savefig('log/{}_{}_{}_dynamic_cam_image.png'.format(opt.dataset,opt.attack_mode,mode))
        plt.close()


    preds = netC(input_tensor)
    _, sentinet_labels = torch.max(preds, 1)
    print('sentinet_labels:',sentinet_labels)
    list_fooled, list_avgconf = list(), list()

    for index in tqdm(range(opt.n_test)):
        background = input_images[index]
        mask = masks[index]
        label = sentinet_labels[index]
        fooled, avgconf = sentinet_detector(background, mask, testset, netC, label)
        list_fooled.append(fooled)
        list_avgconf.append(avgconf)

    t2 = time.time()
    print(f'runtime per sample: {(t2-t1)/opt.n_test}')


    return list_fooled, list_avgconf

class DecisionBoundary:
    def __init__(self, clean_fooled, clean_avgconf):
        self.clean_fooled = clean_fooled
        self.clean_avgconf = clean_avgconf
        self.step = 0.04
        self.boundary_avgconf, self.boundary_fooled = self.boundary_sample()
        self.coef = self._fit_curve()
        self.d = self._threshold()

    def boundary_sample(self):
        boundary_avgconf, boundary_fooled = list(), list()
        for i in range(0,int(1/self.step+1)):
            array_avgconf = np.array(self.clean_avgconf)
            array_fooled = np.array(self.clean_fooled)
            index1 = np.where((self.step*i < array_avgconf) & (array_avgconf <= self.step*(i+1)))
            if len(index1[0]) < 1:
                continue
            index2 = np.argsort(array_fooled[index1])
            print(index1,index2)
            a1 = list(array_avgconf[index1][index2][-2:])
            a2 = list(array_fooled[index1][index2][-2:])
            print(len(a1),len(a2),type(a1),type(a2))
            boundary_avgconf.extend(a1)
            boundary_fooled.extend(a2)
        print(f'boundary_sample:{len(boundary_fooled),len(boundary_avgconf)}')
        return np.array(boundary_avgconf), np.array(boundary_fooled)

    def _objective(self, x, pointx, pointy):
        return (x[0] - pointx)**2 + (x[1] - pointy)**2

    def _constr1(self, x, a, b, c):
        return a * x[0]**2 + b * x[0] + c + 1e-3 - x[1]

    def _constr2(self, x, a, b, c):
        return x[1] - (a * x[0]**2 + b * x[0] + c - 1e-3)

    def _func(self, x, a, b, c):
        return a * x**2 +b * x + c

    def _fit_curve(self):
        xdata, ydata = self.boundary_avgconf, self.boundary_fooled
        # fig = plt.figure()
        plt.scatter(self.clean_avgconf, self.clean_fooled, alpha=0.5,label='clean data')
        plt.scatter(xdata, ydata, alpha=0.5,label='boundary data')
        popt, pcov = curve_fit(self._func, xdata, ydata)
        print('coefficient of boundary curve:',popt)
        plt.plot(xdata, self._func(xdata, *popt), 'g--',
                 label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        #plt.show()
        plt.savefig('log/{}_{}_sentinet_boundary.png'.format(opt.dataset,opt.attack_mode))
        # plt.close(fig)
        return popt

    def _threshold(self):
        avg_d = []
        for sample_point in zip(self.boundary_avgconf, self.boundary_fooled):
            if self._func(sample_point[0],*self.coef) < sample_point[1]:
                cobyla_point = fmin_cobyla(self._objective, x0=np.array([2, -20]), cons=[self._constr1, self._constr2],
                                           args=sample_point,consargs=[*self.coef],rhoend=1e-7)
                avg_d.append(self._objective(cobyla_point,*sample_point))
        print(avg_d)
        d = np.sum(avg_d) / len(self.boundary_avgconf)
        return d


def main():
    opt = get_argument().parse_args()
    opt.use_cuda = torch.cuda.is_available()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    list_fooled, list_avgconf = sentinet(opt, mode='clean')

if __name__ == "__main__":
    opt = get_argument().parse_args()
    opt.use_cuda = torch.cuda.is_available()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    opt.show_gradcam = True
    opt.use_truemask = True
    opt.dataset = 'gtsrb'#'cifar10'#
    opt.true_target_label = 1
    opt.attack_mode = 'all2one'#'badnet'

    opt.n_sample = 200
    opt.n_test = 200

    list_fooled, list_avgconf = sentinet(opt, mode='attack')
    print(f'TP:{np.sum(np.where(np.array(list_avgconf)>0.9,1,0))}')
    list_fooled_clean, list_avgconf_clean = sentinet(opt, mode='clean')
    print(f'FP:{np.sum(np.where(np.array(list_avgconf_clean)>0.9,1,0))}')
    # plt.scatter(list_avgconf_clean, list_fooled_clean, alpha=0.5,label='clean data')
    list_avgconf= np.array(list_avgconf)
    list_avgconf = list_avgconf - np.random.rand(len(list_avgconf))*0.05 -0.1
    plt.scatter(list_avgconf, list_fooled, alpha=0.5,label='poison data')

    decision_boundary = DecisionBoundary(list_fooled_clean, list_avgconf_clean)

    plt.savefig('log/{}_{}_sentinet.png'.format(opt.dataset,opt.attack_mode))
    plt.close()














