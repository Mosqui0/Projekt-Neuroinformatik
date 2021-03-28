import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision

import os, argparse
from tqdm import tqdm
import cv2

from PixelwiseRegression.model import PixelwiseRegression
import PixelwiseRegression.datasets
from PixelwiseRegression.utils import load_model, recover_uvd, select_gpus
from PixelwiseRegression.classifyASL import *

def draw_skeleton(_img, joints, *, output_size=512, rP = 8, linewidth = 4, draw=False, skeleton_mode=0):
    fig, axes = plt.subplots(figsize=(4, 4))
    if joints.shape[0] == 14:

        Index = [13, 1, 0]
        Mid = [13, 3, 2]
        Ring = [13, 5, 4]
        Small = [13, 7, 6]
        Thumb = [13, 10, 9, 8]
        PALM = [11, 13, 12]
        config = [Thumb, Index, Mid, Ring, Small, PALM]
        img = cv2.resize(_img, (output_size, output_size))
        #img3D is a copy to draw the joints
        img3D = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(3):
            img3D[:, :, i] = img
        is_hand = img3D != 0
        #np.max returns max of the given axis
        img3D = img3D / np.max(img3D)
        # img3D = img3D * 0.5 + 0.25
        img3D = 1 - img3D
        img3D[is_hand] *= 0.5
        joints = joints * (img.shape[0] - 1) + np.array([img.shape[0] // 2, img.shape[0] // 2])
        _joint = [(int(joints[i][0]), int(joints[i][1])) for i in range(joints.shape[0])]
        colors = [(1, 0, 0), (0.5, 0.5, 0), (0, 1, 0), (0, 0.5, 0.5), (0, 0, 1), (0.5, 0.5, 0.5)]
        for i in range(6):
            for index in config[i]:
                cv2.circle(img3D, _joint[index], rP, colors[i], -1)
            for j in range(len(config[i]) - 1):
                cv2.line(img3D, _joint[config[i][j]], _joint[config[i][j+1]], colors[i], linewidth)
        if draw:
            axes.imshow(img3D)
            axes.axis("off")
            plt.show()
        else:
            return img3D, xyz_cords
    else:
        if joints.shape[0] == 21:
            if skeleton_mode == 0:
                Index = [0, 1, 2, 3, 4]
                Mid = [0, 5, 6, 7, 8]
                Ring = [0, 9, 10, 11, 12]
                Small = [0, 13, 14, 15, 16]
                Thumb = [0, 17, 18, 19, 20]
                config = [Thumb, Index, Mid, Ring, Small]
            elif skeleton_mode == 1:
                Index = [0, 2, 9, 10, 11]
                Mid = [0, 3, 12, 13, 14]
                Ring = [0, 4, 15, 16, 17]
                Small = [0, 5, 18, 19, 20]
                Thumb = [0, 1, 6, 7, 8]
                config = [Thumb, Index, Mid, Ring, Small]
        elif joints.shape[0] == 16:
        
            Index = [0, 4, 5, 6]
            Mid = [0, 7, 8, 9]
            Ring = [0, 10, 11, 12]
            Small = [0, 13, 14, 15]
            Thumb = [0, 1, 2, 3]
            config = [Thumb, Index, Mid, Ring, Small]
        img = cv2.resize(_img, (output_size, output_size))
        img3D = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(3):
            img3D[:, :, i] = img
        is_hand = img3D != 0
        img3D = img3D / np.max(img3D)
        # img3D = img3D * 0.5 + 0.25
        img3D = 1 - img3D
        img3D[is_hand] *= 0.5
        #Denormalize the Uv coordinates
        #print("Before: ",  joints)
        joints = joints * (img.shape[0] - 1) + np.array([img.shape[0] // 2, img.shape[0] // 2])
        #convert to xy coordinates
        _joint = [(int(joints[i][0]), int(joints[i][1])) for i in range(joints.shape[0])]
        colors = [(1, 0, 0), (0.5, 0.5, 0), (0, 1, 0), (0, 0.5, 0.5), (0, 0, 1)]
        for i in range(5):
            for index in config[i]:
                cv2.circle(img3D, _joint[index], rP, colors[i], -1)
            for j in range(len(config[i]) - 1):
                cv2.line(img3D, _joint[config[i][j]], _joint[config[i][j+1]], colors[i], linewidth)
        if draw:
            axes.imshow(img3D)
            axes.axis("off")
            plt.show()
        else:
            return img3D, _joint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default="default",
        help="the suffix of model file and log file"
    )

    parser.add_argument('--dataset', type=str, default='MSRA', 
        help="choose from MSRA, ICVL, NYU, HAND17"    
    )

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--label_size', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--sigmoid', type=float, default=1.5)
    parser.add_argument('--norm_method', type=str, default='instance', help='choose from batch and instance')
    parser.add_argument('--heatmap_method', type=str, default='softmax', help='choose from softmax and sumz')

    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument('--stages', type=int, default=2)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--level', type=int, default=4)
    parser.add_argument('--seed', type=str, default='final')

    args = parser.parse_args()

    if not os.path.exists("skeleton"):
        os.mkdir("skeleton")

    if not os.path.exists(os.path.join("skeleton", args.dataset)):
        os.mkdir(os.path.join("skeleton", args.dataset))
        os.mkdir(os.path.join("skeleton", args.dataset, "predict"))
        os.mkdir(os.path.join("skeleton", args.dataset, "gt"))

    cv2.namedWindow("predict", 0)
    cv2.namedWindow("ground_truth", 0)

    assert os.path.exists('Model'), "Please put the models in ./Model folder"

    dataset_parameters = {
        "image_size" : args.label_size * 2,
        "label_size" : args.label_size,
        "kernel_size" : args.kernel_size,
        "sigmoid" : args.sigmoid,
        "dataset" : "val",
        "test_only" : False,
    }

    test_loader_parameters = {
        "batch_size" : args.batch_size,
        "shuffle" : True,
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
    }

    model_name = "{}_{}_{}.pt".format(args.dataset, args.suffix, args.seed)

    Dataset = getattr(datasets, "{}Dataset".format(args.dataset))
    testset = Dataset(**dataset_parameters)

    joints = testset.joint_number
    config = testset.config
    threshold = testset.cube_size
    skeleton_mode = 1 if args.dataset == 'HAND17' else 0

    test_loader = torch.utils.data.DataLoader(testset, **test_loader_parameters)

    select_gpus(args.gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = PixelwiseRegression(joints, **model_parameters)
    load_model(model, os.path.join('Model', model_name), eval_mode=True)
    model = model.to(device)

    index = 0
    for batch in iter(test_loader):
        img, label_img, mask, box_size, cube_size, com, uvd, heatmaps, depthmaps = batch
        
        img = img.to(device, non_blocking=True)
        label_img = label_img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        results = model(img, label_img, mask)

        _heatmaps, _depthmaps, _uvd = results[-1]
        #print("UVD SHApe: {}".format(_uvd.shape))
        _uvd = _uvd.detach().cpu().numpy()
        img = img.cpu().numpy()
        uvd = uvd.numpy()
        
        ''' Predicts the ASL sign '''
        #Denormalize the Uv coordinates
        joints44 = _uvd[0,:,:2] * (512 - 1) + np.array([512 // 2, 512 // 2])
        #convert to xy or int? coordinates
        _joint44 = [(int(joints44[i][0]), int(joints44[i][1])) for i in range(joints44.shape[0])]
        palmRadius = calcPalm(_joint44)
        fingers = fingerStretched(_joint44, palmRadius)
        prediction = classifyHandSign(fingers)
        print("Prediction: ", prediction)
        ''' -----------------------'''    
        
        #print("Joints are: ", _uvd[0,:,2])
        skeleton_gt,xyz_gt = draw_skeleton(img[0,0], uvd[0,:,:2], skeleton_mode=skeleton_mode)
        #print(xyz_pre)
        skeleton_pre, xyz_pre = draw_skeleton(img[0,0], _uvd[0,:,:2], skeleton_mode=skeleton_mode)
        skeleton_gt = np.clip(skeleton_gt, 0, 1)
        skeleton_pre = np.clip(skeleton_pre, 0, 1)
        
        #Subtract predict from ground truth
        difference = []
        for i in range(len(xyz_gt)):
            tmpPre = xyz_pre[i]
            tmpgrTr = xyz_gt[i]
            x = tmpPre[0]-tmpgrTr[0]
            y = tmpPre[1]-tmpgrTr[1]
            difference.append((x,y))
        #print("The difference is: ", difference)
        

        cv2.imshow("predict", skeleton_pre)
        cv2.moveWindow("predict", 50,50)
        cv2.imshow("ground_truth", skeleton_gt)
        cv2.moveWindow("ground_truth", 530,50)
        print("Waiting for key to Press: s for saving the images, q for quitting (Press command with images as scene).")
        ch = cv2.waitKey(0)
        if ch == ord('s'):
            cv2.destroyAllWindows()
            plt.imsave(os.path.join("skeleton", args.dataset, "predict", "{}.jpg".format(index)), skeleton_pre)
            plt.imsave(os.path.join("skeleton", args.dataset, "gt", "{}.jpg".format(index)), skeleton_gt)
            index += 1
        elif ch == ord('q'):
            break
