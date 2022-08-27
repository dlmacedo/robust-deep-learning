from __future__ import print_function
import argparse
import torch
import models
import os
import robust_deep_learning as rdl
import data_loader
from torchvision import transforms
import timm

parser = argparse.ArgumentParser(description='detector')
parser.add_argument('-bs', '--batch-size', type=int, default=64, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | tinyimagenet')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--model', required=True, help='resnet | wideresnet')
parser.add_argument('--gpu', type=int, default=None, help='gpu index')
parser.add_argument('--loss', required=True, help='the loss used')
parser.add_argument('--dir', default="", type=str, help='Part of the dir to use')
parser.add_argument('-x', '--executions', default=1, type=int, metavar='N', help='Number of executions (default: 1)')
args = parser.parse_args()

print("\n\n\n\n\n\n\n\n")
print("#############################")
print("#############################")
print("######### DETECTION #########")
print("#############################")
print("#############################")
print(args)

dir_path = os.path.join("experiments", args.dir, "data~"+args.dataset+"+model~"+args.model+"+loss~"+str(args.loss))
file_path = os.path.join(dir_path, "results_ood.csv")

with open(file_path, "w") as results_file:
    results_file.write("EXECUTION,MODEL,IN-DATA,OUT-DATA,LOSS,SCORE,TPR,AUROC,AUPRIN,AUPROUT\n")

args_outf = os.path.join("temp", "ood", args.loss, args.model + '+' + args.dataset)
if os.path.isdir(args_outf) == False:
    os.makedirs(args_outf)

if args.dataset == 'cifar10':
    args.num_classes = 10
    out_dist_list = ['cifar100', 'imagenet_resize', 'lsun_resize', 'svhn']
    in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))])
elif args.dataset == 'cifar100':
    args.num_classes = 100
    out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize', 'svhn']
    in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))])
elif args.dataset == 'imagenet1k':
    args.num_classes = 1000
    out_dist_list = ['imagenet-o']
    args.dataroot = '/mnt/ssd/imagenet1k'
    args.input_size = 224
    args.DEFAULT_CROP_RATIO = 0.875
    in_transform = transforms.Compose([
        transforms.Resize(int(args.input_size / args.DEFAULT_CROP_RATIO)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

for args.execution in range(1, args.executions + 1):    
    print("\n\n\n\nEXECUTION:", args.execution)
    pre_trained_net = os.path.join(dir_path, "model" + str(args.execution) + ".pth")

    if args.loss.split("_")[0] == "softmax":
        args.loss_first_part = rdl.SoftMaxLossFirstPart
        scores = ["MPS"]
    elif args.loss.split("_")[0] == "isomax":
        args.loss_first_part = rdl.IsoMaxLossFirstPart
        scores = ["ES"]
    elif args.loss.split("_")[0] == "isomaxplus":
        args.loss_first_part = rdl.IsoMaxPlusLossFirstPart
        scores = ["MDS"]
    elif args.loss.split("_")[0] == "dismax":
        args.loss_first_part = rdl.DisMaxLossFirstPart
        scores = ["MMLES","MPS"]

    # load networks
    if args.model == 'resnet34':
        model = models.ResNet34(num_c=args.num_classes)
        model.classifier = args.loss_first_part(model.classifier.in_features, model.classifier.out_features)
    elif args.model == 'densenetbc100':
        model = models.DenseNet3(100, int(args.num_classes))
        model.classifier = args.loss_first_part(model.classifier.in_features, model.classifier.out_features)
    elif args.model == "wideresnet2810":
        model = models.Wide_ResNet(depth=28, widen_factor=10, num_classes=args.num_classes)
        model.classifier = args.loss_first_part(model.classifier.in_features, model.classifier.out_features)
    elif args.model == "resnet18":
        model = timm.create_model('resnet18', pretrained=False)
        num_in_features = model.get_classifier().in_features
        model.fc = args.loss_first_part(num_in_features, args.num_classes)

    if args.gpu is not None:
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
    else:
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda"))
    model.cuda(args.gpu)
    model.eval()
    print('load model: ' + args.model)
    
    # load dataset
    print('load target valid data: ', args.dataset)
    _, test_loader = data_loader.getTargetDataSet(args, args.dataset, args.batch_size, in_transform, args.dataroot)

    for score in scores:
        print("\n\n")
        print("###############################")
        print("###############################")
        print("SCORE:", score)
        print("###############################")
        print("###############################")
        base_line_list = []
        print("In-distribution")
        results = rdl.get_outputs_labels_and_metrics(model, test_loader, gpu=args.gpu)
        in_data_scores = rdl.get_scores(results["outputs"], score)

        for out_dist in out_dist_list:
            print('\nOut-distribution: ' + out_dist)
            out_test_loader = data_loader.getNonTargetDataSet(args, out_dist, args.batch_size, in_transform, args.dataroot)
            results = rdl.get_outputs_labels_and_metrics(model, out_test_loader, gpu=args.gpu)
            out_data_scores = rdl.get_scores(results["outputs"], score)
            results = rdl.get_ood_metrics_from_scores(in_data_scores, out_data_scores, fpr=0.05)
            with open(file_path, "a") as results_file:
                results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                    str(args.execution), args.model, args.dataset, out_dist,str(args.loss), score,
                    '{:.2f}'.format(100.*results['tpr_at_given_fpr']),
                    '{:.2f}'.format(100.*results['auroc']),
                    '{:.2f}'.format(100.*results['auprin']),
                    '{:.2f}'.format(100.*results['auprout'])))
