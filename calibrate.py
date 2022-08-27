from __future__ import print_function
import argparse
import torch
import models
import os
import robust_deep_learning as rdl
import loaders
import torch


parser = argparse.ArgumentParser(description='calibrator')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | imagenet1k')
parser.add_argument('--model', required=True, help='model to use')
parser.add_argument('--gpu', type=int, default=None, help='gpu index')
parser.add_argument('--loss', required=True, help='the loss used')
parser.add_argument('--dir', default="", type=str, help='Part of the dir to use')
parser.add_argument('-x', '--executions', default=1, type=int, metavar='N', help='Number of executions (default: 1)')
args = parser.parse_args()

print("\n\n\n\n\n\n\n\n")
print("###############################")
print("###############################")
print("######### CALIBRATION #########")
print("###############################")
print("###############################")
print(args)

dir_path = os.path.join("experiments", args.dir, "data~"+args.dataset+"+model~"+args.model+"+loss~"+str(args.loss))
file_path = os.path.join(dir_path, "results_calib.csv")

with open(file_path, "w") as results_file:
    results_file.write("EXECUTION,MODEL,LOSS,DATA,INFERENCE,OPTIMIZED_METRIC,TEMPERATURE,CALCULATED_METRIC,VALUE\n")

if args.dataset == 'cifar10':
    args.original_dataset = 'cifar10'
    inferences = ["cifar10"]
    args.num_classes = 10
elif args.dataset == 'cifar100':
    args.original_dataset = 'cifar100'
    inferences = ["cifar100"]
    args.num_classes = 100

for args.execution in range(1, args.executions + 1):
    print("\n\n\n\nEXECUTION:", args.execution)
    pre_trained_net = os.path.join(dir_path, "model" + str(args.execution) + ".pth")

    if args.loss.split("_")[0] == "softmax":
        args.loss_first_part = rdl.SoftMaxLossFirstPart
    elif args.loss.split("_")[0] == "isomax":
        args.loss_first_part = rdl.IsoMaxLossFirstPart
    elif args.loss.split("_")[0] == "isomaxplus":
        args.loss_first_part = rdl.IsoMaxPlusLossFirstPart
    elif args.loss.split("_")[0] == "dismax":
        args.loss_first_part = rdl.DisMaxLossFirstPart

    # load networks
    if args.model == 'resnet34':
        model = models.ResNet34(num_c=args.num_classes)
        model.classifier = args.loss_first_part(model.classifier.in_features, model.classifier.out_features)
        model_last_layer = model.classifier
    elif args.model == 'densenetbc100':
        model = models.DenseNet3(100, int(args.num_classes))
        model.classifier = args.loss_first_part(model.classifier.in_features, model.classifier.out_features)
        model_last_layer = model.classifier
    elif args.model == "wideresnet2810":
        model = models.Wide_ResNet(depth=28, widen_factor=10, num_classes=args.num_classes)
        model.linear = args.loss_first_part(model.classifier.in_features, model.classifier.out_features)
        model_last_layer = model.linear
    if args.gpu is not None:
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
    else:
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda"))
    model.cuda(args.gpu)
    model.eval()
    print('load model: ' + args.model)
    
    for metric_to_optimize in ["ECE"]:
        model_last_layer.temperature.data = torch.tensor([1.0]).cuda(args.gpu)
        print("\n#########################")
        args.dataset = args.original_dataset
        print(args.dataset.upper())
        print("#########################")
        image_loaders = loaders.ImageLoader(args)
        _, _, in_data_valid_loader = image_loaders.get_loaders()
        rdl.calibrate_temperature(model_last_layer, model, in_data_valid_loader, optimize="ECE", gpu=args.gpu)

        for inference in inferences:
            print("\n#########################")
            args.dataset = inference
            print(args.dataset.upper())
            print("#########################")
            image_loaders = loaders.ImageLoader(args)
            _, _, in_data_valid_loader = image_loaders.get_loaders()
            results = rdl.get_outputs_labels_and_metrics(model, in_data_valid_loader, gpu=args.gpu)

            if 0.001 < model_last_layer.temperature.item() < 100:
                with open(file_path, "a") as results_file:
                    results_file.write("{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.model, str(args.loss), args.original_dataset, args.dataset,
                        metric_to_optimize, model_last_layer.temperature.item(), "ECE", results["ece"]))
                with open(file_path, "a") as results_file:
                    results_file.write("{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.model, str(args.loss), args.original_dataset, args.dataset,
                        metric_to_optimize, model_last_layer.temperature.item(), "NLL", results["nll"]))
                with open(file_path, "a") as results_file:
                    results_file.write("{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.model, str(args.loss), args.original_dataset, args.dataset,
                        metric_to_optimize, model_last_layer.temperature.item(), "ACC", results["acc"]))
