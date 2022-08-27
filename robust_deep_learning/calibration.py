from .general import get_outputs_labels_and_metrics
from .uncertainty import ECELoss
import scipy.optimize as opt
import torch
import numpy


def calibrate_temperature(model_last_layer, model, loader, optimize="ECE", gpu=None):
    print("Calibrating...")
    results = get_outputs_labels_and_metrics(model, loader, gpu=gpu)

    if optimize == "ECE":
        ece_criterion = ECELoss()  
        def ece_eval(temperature):
            loss = ece_criterion.loss(results["outputs"].cpu().numpy()/temperature, results["labels"].cpu().numpy(),15)
            return loss
        temperature_for_min, min_value, _ = opt.fmin_l_bfgs_b(ece_eval, numpy.array([1.0]), approx_grad=True, bounds=[(0.001,100)])
        temperature_for_min = temperature_for_min[0]
        model_last_layer.temperature.data = torch.tensor([temperature_for_min]).to(torch.device('cuda')) 
    elif optimize == "NONE":
        temperature_for_min = 1
        model_last_layer.temperature = temperature_for_min
    print("Calibrated!!!")
