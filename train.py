import sys
import argparse
import os
import random
import pandas
import torch
import utils
import torchvision
import models
import loaders
import robust_deep_learning as rdl
import statistics
import math
import torchnet
import numpy
import timm

numpy.set_printoptions(edgeitems=5, linewidth=160, formatter={'float': '{:0.6f}'.format})
torch.set_printoptions(edgeitems=5, precision=6, linewidth=160)
pandas.options.display.float_format = '{:,.6f}'.format
pandas.set_option('display.width', 160)

parser = argparse.ArgumentParser(description='trainer')
parser.add_argument('-x', '--executions', default=1, type=int, metavar='N', help='Number of executions (default: 1)')
parser.add_argument('-e', '--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-bs', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-lr', '--original-learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-lrdr', '--learning-rate-decay-rate', default=0.1, type=float, metavar='LRDR', help='learning rate decay rate')
parser.add_argument('-lrde', '--learning-rate-decay-epochs', default="150 200 250", metavar='LRDE', help='learning rate decay epochs')
parser.add_argument('-lrdp', '--learning-rate-decay-period', default=500, type=int, metavar='LRDP', help='learning rate decay period')
parser.add_argument('-mm', '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('-wd', '--weight-decay', default=1*1e-4, type=float, metavar='W', help='weight decay (default: 1*1e-4)')
parser.add_argument('-pf', '--print-freq', default=1, type=int, metavar='N', help='print frequency (default: 1)')
parser.add_argument('--gpu', default=None, type=int, help='id for CUDA_VISIBLE_DEVICES')
parser.add_argument('--dir', default="all", type=str, metavar='PATHS', help='relative paths for the experiments')
parser.add_argument('-sd', '--seed', default=42, type=int, metavar='N', help='Seed (default: 42)')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | tinyimagenet')
parser.add_argument('--model', required=True, help='the model to be used')
parser.add_argument('--loss', required=True, help='the loss to be used')
args = parser.parse_args()
args.learning_rate_decay_epochs = [int(item) for item in args.learning_rate_decay_epochs.split()]

print("\n\n\n\n\n\n\n\n")
print("***************************************************************")
print("***************************************************************")
print("***************************************************************")
print("***************************************************************")
print("***************************************************************")

random.seed(args.seed)
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print("seed", args.seed)

torch.backends.cudnn.benchmark = True
if args.executions == 1:
    torch.backends.cudnn.deterministic = True
    print("Deterministic!!!")
else:
    torch.backends.cudnn.deterministic = False
    print("No deterministic!!!")

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__Number CUDA Devices:', torch.cuda.device_count())
print('Active CUDA Device: GPU', torch.cuda.current_device())


def train_epoch():
    print()
    model.train()

    loss_meter = utils.MeanMeter()
    accuracy_meter = torchnet.meter.ClassErrorMeter(topk=[1], accuracy=True)
    epoch_logits = {"intra": [], "inter": []}
    epoch_metrics = {"max_probs": [], "entropies": [], "max_logits": [], "mean_logits": []}

    for batch_index, batch_data in enumerate(in_data_train_loader):
        batch_index += 1

        inputs = batch_data[0]
        targets = batch_data[1]
        inputs = inputs.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)

        inputs, targets = criterion.preprocess(inputs, targets)                      

        outputs = model(inputs)
        loss, scale, inter_logits, intra_logits = criterion(outputs, targets)

        max_logits = outputs.max(dim=1)[0]
        mean_logits = outputs.mean(dim=1)
        probabilities = torch.nn.Softmax(dim=1)(outputs)
        max_probs = probabilities.max(dim=1)[0]
        entropies = utils.entropies_from_probabilities(probabilities)

        loss_meter.add(loss.item(), targets.size(0))
        accuracy_meter.add(outputs.detach(), targets.detach())

        intra_logits = intra_logits.tolist()
        inter_logits = inter_logits.tolist()
        if args.number_of_model_classes > 100:
            epoch_logits["intra"] = intra_logits
            epoch_logits["inter"] = inter_logits
        else:
            epoch_logits["intra"] += intra_logits
            epoch_logits["inter"] += inter_logits
        epoch_metrics["max_probs"] += max_probs.tolist()
        epoch_metrics["max_logits"] += max_logits.tolist()
        epoch_metrics["mean_logits"] += mean_logits.tolist()
        epoch_metrics["entropies"] += (entropies/math.log(args.number_of_model_classes)).tolist()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % args.print_freq == 0:
            print('Train Epoch: [{0}][{1:3}/{2}]\t'
                    'Loss {loss:.8f}\t\t'
                    'Acc1 {acc1_meter:.2f}\t'
                    'IELM {inter_logits_mean:.4f}\t'
                    'IELS {inter_logits_std:.8f}\t\t'
                    'IALM {intra_logits_mean:.4f}\t'
                    'IALS {intra_logits_std:.8f}'
                    .format(epoch, batch_index, len(in_data_train_loader),
                            loss=loss_meter.avg,
                            acc1_meter=accuracy_meter.value()[0],
                            inter_logits_mean=statistics.mean(inter_logits),
                            inter_logits_std=statistics.stdev(inter_logits),
                            intra_logits_mean=statistics.mean(intra_logits),
                            intra_logits_std=statistics.stdev(intra_logits)))

    print('\n#### TRAIN ACC1:\t{0:.4f}\n'.format(accuracy_meter.value()[0]))
    return loss_meter.avg, accuracy_meter.value()[0], scale, epoch_logits, epoch_metrics


def validate_epoch():
    print()
    model.eval()

    loss_meter = utils.MeanMeter()
    accuracy_meter = torchnet.meter.ClassErrorMeter(topk=[1], accuracy=True)
    epoch_logits = {"intra": [], "inter": []}
    epoch_metrics = {"max_probs": [], "entropies": [], "max_logits": [], "mean_logits": []}

    with torch.no_grad():
        for batch_index, batch_data in enumerate(in_data_valid_loader):
            batch_index += 1

            inputs = batch_data[0]
            targets = batch_data[1]
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

            outputs = model(inputs)

            loss, scale, inter_logits, intra_logits = criterion(outputs, targets)

            max_logits = outputs.max(dim=1)[0]
            mean_logits = outputs.mean(dim=1)
            probabilities = torch.nn.Softmax(dim=1)(outputs)
            max_probs = probabilities.max(dim=1)[0]
            entropies = utils.entropies_from_probabilities(probabilities)

            loss_meter.add(loss.item(), inputs.size(0))
            accuracy_meter.add(outputs.detach(), targets.detach())

            intra_logits = intra_logits.tolist()
            inter_logits = inter_logits.tolist()
            if args.number_of_model_classes > 100:
                epoch_logits["intra"] = intra_logits
                epoch_logits["inter"] = inter_logits
            else:
                epoch_logits["intra"] += intra_logits
                epoch_logits["inter"] += inter_logits
            epoch_metrics["max_probs"] += max_probs.tolist()
            epoch_metrics["max_logits"] += max_logits.tolist()
            epoch_metrics["mean_logits"] += mean_logits.tolist()
            epoch_metrics["entropies"] += (entropies/math.log(args.number_of_model_classes)).tolist()

            if batch_index % args.print_freq == 0:
                print('Valid Epoch: [{0}][{1:3}/{2}]\t'
                        'Loss {loss:.8f}\t\t'
                        'Acc1 {acc1_meter:.2f}\t'
                        'IELM {inter_logits_mean:.4f}\t'
                        'IELS {inter_logits_std:.8f}\t\t'
                        'IALM {intra_logits_mean:.4f}\t'
                        'IALS {intra_logits_std:.8f}'
                        .format(epoch, batch_index, len(in_data_valid_loader),
                                loss=loss_meter.avg,
                                acc1_meter=accuracy_meter.value()[0],
                                inter_logits_mean=statistics.mean(inter_logits),
                                inter_logits_std=statistics.stdev(inter_logits),
                                intra_logits_mean=statistics.mean(intra_logits),
                                intra_logits_std=statistics.stdev(intra_logits)))

    print('\n#### VALID ACC1:\t{0:.4f}\n'.format(accuracy_meter.value()[0]))
    return loss_meter.avg, accuracy_meter.value()[0], scale, epoch_logits, epoch_metrics

print("***************************************************************")
args.relative_path = os.path.join("data~"+args.dataset+"+model~"+args.model+"+loss~"+args.loss)
args.experiment_path = os.path.join("experiments", args.dir, args.relative_path)

if not os.path.exists(args.experiment_path):
    os.makedirs(args.experiment_path)
print("EXPERIMENT PATH:", args.experiment_path)

args.executions_best_results_file_path = os.path.join(args.experiment_path, "results_acc.csv")
args.executions_raw_results_file_path = os.path.join(args.experiment_path, "results_raw.csv")

print("DATASET:", args.dataset)
print("MODEL:", args.model)
print("LOSS:", args.loss)

args.number_of_model_classes = None
if args.dataset == "cifar10":
    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 10
    args.data_type = "image"
elif args.dataset == "cifar100":
    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 100
    args.data_type = "image"
elif args.dataset == "tinyimagenet":
    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 200
    args.data_type = "image"
elif args.dataset == "imagenet1k":
    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 1000
    args.data_type = "image"

if args.model == "resnet18":
    args.input_size = 224
    args.DEFAULT_CROP_RATIO = 0.875
    args.interpolation = torchvision.transforms.functional.InterpolationMode.BILINEAR

if args.loss.split("_")[0] == "softmax":
    args.loss_first_part = rdl.SoftMaxLossFirstPart
    args.loss_second_part = rdl.SoftMaxLossSecondPart
elif args.loss.split("_")[0] == "isomax":
    args.loss_first_part = rdl.IsoMaxLossFirstPart
    args.loss_second_part = rdl.IsoMaxLossSecondPart
elif args.loss.split("_")[0] == "isomaxplus":
    args.loss_first_part = rdl.IsoMaxPlusLossFirstPart
    args.loss_second_part = rdl.IsoMaxPlusLossSecondPart
elif args.loss.split("_")[0] == "dismax":
    args.loss_first_part = rdl.DisMaxLossFirstPart
    args.loss_second_part = rdl.DisMaxLossSecondPart
elif args.loss.split("_")[0] == "dismax2":
    args.loss_first_part = rdl.DisMax2LossFirstPart
    args.loss_second_part = rdl.DisMax2LossSecondPart
else:
    sys.exit('You should pass a valid loss to use!!!')
print("=> creating model '{}'".format(args.model))

image_loaders = loaders.ImageLoader(args)
in_data_train_loader, _, in_data_valid_loader = image_loaders.get_loaders()
print("\nDATASET:", args.dataset)

print("***************************************************************")

for args.execution in range(1, args.executions + 1):
    print("\n\n\n\n################ EXECUTION:", args.execution, "OF", args.executions, "################")
    args.best_model_file_path = os.path.join(args.experiment_path, "model" + str(args.execution) + ".pth")
    utils.save_dict_list_to_csv([vars(args)], args.experiment_path, "args")
    print("\nARGUMENTS:", dict(utils.load_dict_list_from_csv(args.experiment_path, "args")[0]))

    if args.model == "resnet34":
        model = models.ResNet34(num_c=args.number_of_model_classes)
        model.classifier = args.loss_first_part(model.classifier.in_features, model.classifier.out_features)
        model_last_layer = model.classifier
    elif args.model == "densenetbc100":
        model = models.DenseNet3(100, int(args.number_of_model_classes))
        model.classifier = args.loss_first_part(model.classifier.in_features, model.classifier.out_features)
        model_last_layer = model.classifier
    elif args.model == "wideresnet2810":
        model = models.Wide_ResNet(depth=28, widen_factor=10, num_classes=args.number_of_model_classes)
        model.linear = args.loss_first_part(model.linear.in_features, model.linear.out_features)
        model_last_layer = model.linear
    elif args.model == "resnet18":
        model = timm.create_model('resnet18', pretrained=False)
        print(model.default_cfg)
        model.fc = args.loss_first_part(model.get_classifier().in_features, args.number_of_model_classes)
        model_last_layer = model.fc
    model.cuda(args.gpu)
    print("\nMODEL:", model)

    with open(os.path.join(args.experiment_path, 'model.arch'), 'w') as file:
        print(model, file=file)
    print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    utils.print_num_params(model)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    criterion = args.loss_second_part(model_last_layer, debug=True, gpu=args.gpu)   
    if args.loss.split("_")[0].startswith("dismax"):
        if args.loss.split("_")[1].startswith("fpr"):
            criterion = args.loss_second_part(model_last_layer, add_on="fpr", debug=True, gpu=args.gpu)

    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.original_learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.learning_rate_decay_epochs, gamma=args.learning_rate_decay_rate)
    print("\nTRAIN:", criterion, optimizer, scheduler)

    if args.execution == 1:
        with open(args.executions_best_results_file_path, "w") as best_results:
            best_results.write(
                "DATA,MODEL,LOSS,EXECUTION,EPOCH,"
                "TRAIN LOSS,TRAIN ACC1,TRAIN SCALE,"
                "TRAIN INTRA_LOGITS MEAN,TRAIN INTRA_LOGITS STD,TRAIN INTER_LOGITS MEAN,TRAIN INTER_LOGITS STD,"
                "TRAIN MAX_PROBS MEAN,TRAIN MAX_PROBS STD,TRAIN ENTROPIES MEAN,TRAIN ENTROPIES STD,"
                "VALID LOSS,VALID ACC1,VALID SCALE,"
                "VALID INTRA_LOGITS MEAN,VALID INTRA_LOGITS STD,VALID INTER_LOGITS MEAN,VALID INTER_LOGITS STD,"
                "VALID MAX_PROBS MEAN,VALID MAX_PROBS STD,VALID ENTROPIES MEAN,VALID ENTROPIES STD\n")
        with open(args.executions_raw_results_file_path, "w") as raw_results:
            raw_results.write("DATA,MODEL,LOSS,EXECUTION,EPOCH,SET,METRIC,VALUE\n")

    print("\n################ TRAINING AND VALIDATING ################")        
    best_model_results = {"VALID ACC1": 0}

    for epoch in range(1, args.epochs + 1):
        print("\n######## EPOCH:", epoch, "OF", args.epochs, "########")

        for param_group in optimizer.param_groups:
            print("\nLEARNING RATE:\t\t", param_group["lr"])                

        train_loss, train_acc1, train_scale, train_epoch_logits, train_epoch_metrics = train_epoch()           
        valid_loss, valid_acc1, valid_scale, valid_epoch_logits, valid_epoch_metrics = validate_epoch()
        
        if scheduler is not None:
            scheduler.step()

        train_intra_logits_mean = statistics.mean(train_epoch_logits["intra"])
        train_intra_logits_std = statistics.pstdev(train_epoch_logits["intra"])
        train_inter_logits_mean = statistics.mean(train_epoch_logits["inter"])
        train_inter_logits_std = statistics.pstdev(train_epoch_logits["inter"])
        train_max_probs_mean = statistics.mean(train_epoch_metrics["max_probs"])
        train_max_probs_std = statistics.pstdev(train_epoch_metrics["max_probs"])
        train_entropies_mean = statistics.mean(train_epoch_metrics["entropies"])
        train_entropies_std = statistics.pstdev(train_epoch_metrics["entropies"])
        valid_intra_logits_mean = statistics.mean(valid_epoch_logits["intra"])
        valid_intra_logits_std = statistics.pstdev(valid_epoch_logits["intra"])
        valid_inter_logits_mean = statistics.mean(valid_epoch_logits["inter"])
        valid_inter_logits_std = statistics.pstdev(valid_epoch_logits["inter"])
        valid_max_probs_mean = statistics.mean(valid_epoch_metrics["max_probs"])
        valid_max_probs_std = statistics.pstdev(valid_epoch_metrics["max_probs"])
        valid_entropies_mean = statistics.mean(valid_epoch_metrics["entropies"])
        valid_entropies_std = statistics.pstdev(valid_epoch_metrics["entropies"])

        print("\n####################################################")
        print("TRAIN MAX PROB MEAN:\t", train_max_probs_mean)
        print("TRAIN MAX PROB STD:\t", train_max_probs_std)
        print("VALID MAX PROB MEAN:\t", valid_max_probs_mean)
        print("VALID MAX PROB STD:\t", valid_max_probs_std)
        print("####################################################\n")

        print("\n####################################################")
        print("TRAIN ENTROPY MEAN:\t", train_entropies_mean)
        print("TRAIN ENTROPY STD:\t", train_entropies_std)
        print("VALID ENTROPY MEAN:\t", valid_entropies_mean)
        print("VALID ENTROPY STD:\t", valid_entropies_std)
        print("####################################################\n")

        with open(args.executions_raw_results_file_path, "a") as raw_results:
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "TRAIN", "LOSS", train_loss))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "TRAIN", "ACC1", train_acc1))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "TRAIN", "SCALE", train_scale))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "TRAIN", "INTRA_LOGITS MEAN", train_intra_logits_mean))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "TRAIN", "INTRA_LOGITS STD", train_intra_logits_std))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "TRAIN", "INTER_LOGITS MEAN", train_inter_logits_mean))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "TRAIN", "INTER_LOGITS STD", train_inter_logits_std))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "TRAIN", "MAX_PROBS MEAN", train_max_probs_mean))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "TRAIN", "MAX_PROBS STD", train_max_probs_std))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "TRAIN", "ENTROPIES MEAN", train_entropies_mean))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "TRAIN", "ENTROPIES STD", train_entropies_std))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "VALID", "LOSS", valid_loss))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "VALID", "ACC1", valid_acc1))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "VALID", "SCALE", valid_scale))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "VALID", "INTRA_LOGITS MEAN", valid_intra_logits_mean))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "VALID", "INTRA_LOGITS STD", valid_intra_logits_std))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "VALID", "INTER_LOGITS MEAN", valid_inter_logits_mean))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "VALID", "INTER_LOGITS STD", valid_inter_logits_std))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "VALID", "MAX_PROBS MEAN", valid_max_probs_mean))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "VALID", "MAX_PROBS STD", valid_max_probs_std))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "VALID", "ENTROPIES MEAN", valid_entropies_mean))
            raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                args.dataset, args.model, args.loss, args.execution, epoch, "VALID", "ENTROPIES STD", valid_entropies_std))

        print()
        print("TRAIN ==>>\tIELM: {0:.8f}\tIELS: {1:.8f}\tIALM: {2:.8f}\tIALS: {3:.8f}".format(
            train_inter_logits_mean, train_inter_logits_std, train_intra_logits_mean, train_intra_logits_std))
        print("VALID ==>>\tIELM: {0:.8f}\tIELS: {1:.8f}\tIALM: {2:.8f}\tIALS: {3:.8f}".format(
            valid_inter_logits_mean, valid_inter_logits_std, valid_intra_logits_mean, valid_intra_logits_std))
        print()
        print("\nDATA:", args.dataset)
        print("MODEL:", args.model)
        print("LOSS:", args.loss, "\n")

        if valid_acc1 > best_model_results["VALID ACC1"]:
            print("!+NEW BEST MODEL VALID ACC1!")
            best_model_results = {
                "DATA": args.dataset,
                "MODEL": args.model,
                "LOSS": args.loss,
                "EXECUTION": args.execution,
                "EPOCH": epoch,
                "TRAIN LOSS": train_loss,
                "TRAIN ACC1": train_acc1,
                "TRAIN SCALE": train_scale,
                "TRAIN INTRA_LOGITS MEAN": train_intra_logits_mean,
                "TRAIN INTRA_LOGITS STD": train_intra_logits_std,
                "TRAIN INTER_LOGITS MEAN": train_inter_logits_mean,
                "TRAIN INTER_LOGITS STD": train_inter_logits_std,
                "TRAIN MAX_PROBS MEAN": train_max_probs_mean,
                "TRAIN MAX_PROBS STD": train_max_probs_std,
                "TRAIN ENTROPIES MEAN": train_entropies_mean,
                "TRAIN ENTROPIES STD": train_entropies_std,
                "VALID LOSS": valid_loss,
                "VALID ACC1": valid_acc1,
                "VALID SCALE": valid_scale,
                "VALID INTRA_LOGITS MEAN": valid_intra_logits_mean,
                "VALID INTRA_LOGITS STD": valid_intra_logits_std,
                "VALID INTER_LOGITS MEAN": valid_inter_logits_mean,
                "VALID INTER_LOGITS STD": valid_inter_logits_std,
                "VALID MAX_PROBS MEAN": valid_max_probs_mean,
                "VALID MAX_PROBS STD": valid_max_probs_std,
                "VALID ENTROPIES MEAN": valid_entropies_mean,
                "VALID ENTROPIES STD": valid_entropies_std,}

            print("!+NEW BEST MODEL VALID ACC1:\t\t{0:.4f} IN EPOCH {1}! SAVING {2}\n".format(valid_acc1, epoch, args.best_model_file_path))
            torch.save(model.state_dict(), args.best_model_file_path)
            numpy.save(os.path.join(args.experiment_path, "best_model"+str(args.execution)+"_train_epoch_logits.npy"), train_epoch_logits)
            numpy.save(os.path.join(args.experiment_path, "best_model"+str(args.execution)+"_train_epoch_metrics.npy"), train_epoch_metrics)
            numpy.save(os.path.join(args.experiment_path, "best_model"+str(args.execution)+"_valid_epoch_logits.npy"), valid_epoch_logits)
            numpy.save(os.path.join(args.experiment_path, "best_model"+str(args.execution)+"_valid_epoch_metrics.npy"), valid_epoch_metrics)

        print('!$$$$ BEST MODEL TRAIN ACC1:\t\t{0:.4f}'.format(best_model_results["TRAIN ACC1"]))
        print('!$$$$ BEST MODEL VALID ACC1:\t\t{0:.4f}'.format(best_model_results["VALID ACC1"]))

    with open(args.executions_best_results_file_path, "a") as best_results:
        best_results.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            best_model_results["DATA"],
            best_model_results["MODEL"],
            best_model_results["LOSS"],
            best_model_results["EXECUTION"],
            best_model_results["EPOCH"],
            best_model_results["TRAIN LOSS"],
            best_model_results["TRAIN ACC1"],
            best_model_results["TRAIN SCALE"],
            best_model_results["TRAIN INTRA_LOGITS MEAN"],
            best_model_results["TRAIN INTRA_LOGITS STD"],
            best_model_results["TRAIN INTER_LOGITS MEAN"],
            best_model_results["TRAIN INTER_LOGITS STD"],
            best_model_results["TRAIN MAX_PROBS MEAN"],
            best_model_results["TRAIN MAX_PROBS STD"],
            best_model_results["TRAIN ENTROPIES MEAN"],
            best_model_results["TRAIN ENTROPIES STD"],
            best_model_results["VALID LOSS"],
            best_model_results["VALID ACC1"],
            best_model_results["VALID SCALE"],
            best_model_results["VALID INTRA_LOGITS MEAN"],
            best_model_results["VALID INTRA_LOGITS STD"],
            best_model_results["VALID INTER_LOGITS MEAN"],
            best_model_results["VALID INTER_LOGITS STD"],
            best_model_results["VALID MAX_PROBS MEAN"],
            best_model_results["VALID MAX_PROBS STD"],
            best_model_results["VALID ENTROPIES MEAN"],
            best_model_results["VALID ENTROPIES STD"],))

experiment_results = pandas.read_csv(os.path.join(os.path.join(args.experiment_path, "results_acc.csv")))
print("\n################################\n", "EXPERIMENT RESULTS", "\n################################")
print(args.experiment_path)
print("\n", experiment_results.transpose())
print()
