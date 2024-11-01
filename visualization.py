# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.ndimage
# import pandas as pde

# plt.style.use("fivethirtyeight")
# plt.figure(figsize=(10, 7))
# # plt.figure(dpi=100,figsize=(5,3)) # 分辨率参数-dpi，画布大小参数-figsize
# data = pd.read_csv('./messidor/baseline/performance.csv')

# xdata = data.iloc[:, data.columns.get_loc('Training dataset size')]
# ydata = data.iloc[:, data.columns.get_loc('Accuracy')]

# plt.plot(xdata,ydata,'cornflowerblue',label='performance',linewidth=1)

# # orangered
# my_x_ticks = np.arange(100, 768, 100)
# plt.xticks(my_x_ticks)
# my_y_ticks = np.arange(0.64, 0.9, 0.04)
# plt.yticks(my_y_ticks)

# plt.title("Messidor",size=15)

# plt.xlabel('Training dataset size',size=12)
# plt.ylabel('Testing Accuracy',size=12)
# plt.axhline(y=0.8667, color = 'grey', linewidth=1.5, linestyle="-", label='Inception v3' )
# plt.legend(frameon=True,borderaxespad=0, prop={'size': 15, "family": "Times New Roman"})
# plt.savefig('./deepfool.jpg', format='jpg',  bbox_inches='tight', transparent=True, dpi=600)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from data import Data
# 假设您的模型定义在YourModelClass中，并且您已经加载了预训练的权重
from nets import MSSEG_model # 请替换为您的模型类
from utils import get_dataset, get_net, get_strategy
from config import parse_args
args = parse_args()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def recompone_overlap(preds,full_imgs_list,test_images,stride=96, image_num=8251):
    patch_w = preds.shape[2] # pred (157341, 64, 64)
    patch_h = preds.shape[1]
#     print(preds.shape)
    final_full_prob = []
    a = 0
    for x in range(image_num):
        input_array = full_imgs_list[x] # 
        # print(input_array.shape)
        ori_image = test_images[x] # 
        # print(ori_image.shape)
        img_h = np.array(input_array).shape[0]
        img_w = np.array(input_array).shape[1]
        
        full_prob = np.zeros((img_h,img_w)) #(352, 160)
        full_sum = np.zeros((img_h,img_w))#

        for i in range((img_h-patch_h)//stride+1):
            for j in range((img_w-patch_w)//stride+1):
                full_prob[i*stride:(i*stride)+patch_h,j*stride:(j*stride)+patch_w]+=preds[a] # Accumulate predicted values
                full_sum[i*stride:(i*stride)+patch_h,j*stride:(j*stride)+patch_w]+=1  # Accumulate the number of predictions
                a+=1
        final_avg = full_prob/full_sum # Take the average
        final_avg = final_avg[0:ori_image.shape[0],0:ori_image.shape[1]]
        # print("final_avg",final_avg.max())
        assert(np.max(final_avg)<=1.0) # max value for a pixel is 1.0
        assert(np.min(final_avg)>=0.0) # min value for a pixel is 0.0
        final_avg[final_avg>=0.5] = 1
        final_avg[final_avg<0.5] = 0
        final_full_prob.append(final_avg)
    return final_full_prob

import matplotlib.pyplot as plt

def visualize_sample(sample, ground_truth, prediction, save_path):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(sample, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='gray')
    plt.title('Our Prediction')
    plt.axis('off')
    
    # Save the figure to the specified path
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 加载数据
X_train, Y_train, X_val, Y_val, X_test, Y_test, handler, full_test_imgs_list, x_test_slice, test_brain_images, test_brain_masks = get_dataset(args.dataset_name,supervised=False)
dataset = Data(X_train, Y_train, X_val, Y_val, X_test, Y_test, handler)

test_dataset = dataset.get_test_data()

clf = MSSEG_model().to(device)
model_path = f'/home/siteng/active_learning_seg/msseg_3/result/AdversarialAttack-42/model_12.pth' # 请替换为正确的路径
# clf.load_state_dict(torch.load(model_path, map_location=device))
torch.load(model_path, map_location=device)

clf.eval()
preds = []
gts = []
loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
with torch.no_grad(): 
    for x, y, idxs in loader:
        x, y = x.unsqueeze(1), y.unsqueeze(1)
        x, y = x.to(device), y.to(device)
        out,_ = clf(x)
        outputs = out.data.cpu().numpy()
        preds.append(outputs)
        gts.append(y.data.cpu().numpy())
ground_truths = np.concatenate(gts, axis=0)#(40617, 1, 128, 128)
ground_truth_patches = np.expand_dims(ground_truths,axis=1)#(40617, 1, 1, 128, 128)
ground_truth_imgs = recompone_overlap(ground_truth_patches.squeeze(), full_test_imgs_list, x_test_slice, stride=96, image_num=8251)

predictions = np.concatenate(preds, axis=0)#(40617, 1, 128, 128)
pred_patches = np.expand_dims(predictions,axis=1)#(40617, 1, 1, 128, 128)
pred_imgs = recompone_overlap(pred_patches.squeeze(), full_test_imgs_list, x_test_slice, stride=96, image_num=8251)


for i in range(len(full_test_imgs_list)):
    if np.sum(ground_truth_imgs[i]) != 0:
        print(x_test_slice[i].shape,ground_truth_imgs[i].shape,pred_imgs[i].shape)
        visualize_sample(np.array(x_test_slice[i], dtype=float), np.array(ground_truth_imgs[i], dtype=float), np.array(pred_imgs[i], dtype=float), f'visualization/Sample {i+1}')



