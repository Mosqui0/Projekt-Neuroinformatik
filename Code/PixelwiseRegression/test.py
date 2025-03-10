import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision

import os, argparse
from tqdm import tqdm

from model import PixelwiseRegression
import datasets
from utils import load_model, recover_uvd, select_gpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default="default",
        help="the suffix of model file and log file"
    )

    parser.add_argument('--dataset', type=str, default='MSRA', 
        help="choose from MSRA, ICVL, NYU, HAND17"    
    )

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--label_size', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--sigmoid', type=float, default=1.5)
    parser.add_argument('--norm_method', type=str, default='instance', help='choose from batch and instance')
    parser.add_argument('--heatmap_method', type=str, default='softmax', help='choose from softmax and sum')
    parser.add_argument('--process_mode', type=str, default='uvd', help='choose from uvd and bb')
    parser.add_argument('--filter_size', type=int, default=3)

    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument("--num_workers", type=int, default=9999)
    parser.add_argument('--stages', type=int, default=2)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--level', type=int, default=4)
    parser.add_argument('--seed', type=str, default='final')

    args = parser.parse_args()

    if not os.path.exists("Result"):
        os.mkdir("Result")

    assert os.path.exists('Model'), "put models in ./Model folder"

    dataset_parameters = {
        "image_size" : args.label_size * 2,
        "label_size" : args.label_size,
        "kernel_size" : args.kernel_size,
        "sigmoid" : args.sigmoid,
        "dataset" : "test",
        "process_mode" : args.process_mode,
        "test_only" : True,
    }

    test_loader_parameters = {
        "batch_size" : args.batch_size,
        "shuffle" : False,
        "pin_memory" : True, 
        "drop_last" : False,
        "num_workers" : min(args.num_workers, os.cpu_count()),
    }

    model_parameters = {
        "stage" : args.stages, 
        "label_size" : args.label_size, 
        "features" : args.features, 
        "level" : args.level,
        "norm_method" : args.norm_method,
        "heatmap_method" : args.heatmap_method,
        "kernel_size" : args.filter_size,
    }

    model_name = "{}_{}_{}.pt".format(args.dataset, args.suffix, args.seed)

    Dataset = getattr(datasets, "{}Dataset".format(args.dataset))
    testset = Dataset(**dataset_parameters)

    joints = testset.joint_number
    config = testset.config
    # threshold = testset.threshold
    threshold = testset.cube_size

    test_loader = torch.utils.data.DataLoader(testset, **test_loader_parameters)

    select_gpus(args.gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = PixelwiseRegression(joints, **model_parameters)
    load_model(model, os.path.join('Model', model_name), eval_mode=True)
    model = model.to(device)

    print("running on test dataset ......")
    with torch.no_grad(), tqdm(total=len(testset) // args.batch_size + 1) as pbar:
        pre_uvd = []
        for batch in iter(test_loader):
            img, label_img, mask, box_size, cube_size, com = batch
            
            img = img.to(device, non_blocking=True)
            label_img = label_img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            results = model(img, label_img, mask)

            _heatmaps, _depthmaps, _uvd = results[-1]

            _uvd = _uvd.cpu()
            _uvd = recover_uvd(_uvd, box_size, com, cube_size)
            _uvd = _uvd.numpy()

            if args.dataset == 'HAND17':
                _uvd = testset.uvd2xyz(_uvd)

            pre_uvd.append(_uvd.reshape(-1, joints * 3))

            pbar.update(1)
        
        pre_uvd = np.concatenate(pre_uvd, axis=0)

        if args.seed == 'final':
            result_name = "Result/{}_{}.txt".format(args.dataset, args.suffix)
        else:
            result_name = "Result/{}_{}_{}.txt".format(args.dataset, args.suffix, args.seed)

        np.savetxt(result_name, pre_uvd, fmt="%.3f")

        if args.dataset == 'HAND17':
            with open(result_name, 'r') as f:
                datatext = f.readlines()

            savetext = []
            for index, text in enumerate(datatext):
                text = text.strip()
                fragment = ['frame\\images\\image_D%08d.png'%(index+1)] + text.split()
                savetext.append("\t".join(fragment))

            with open(result_name, 'w') as f:
                f.write('\n'.join(savetext))
