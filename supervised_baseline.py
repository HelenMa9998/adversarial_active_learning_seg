import torch
from utils import get_dataset, get_net, get_strategy
from data import Data
from config import parse_args

from seed import setup_seed
# fix random seed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

setup_seed(0)
#supervised learning baseline
args = parse_args()
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# get dataset
X_train, Y_train, X_val, Y_val, X_test, Y_test, handler, full_test_imgs_list, x_test_slice, test_brain_images, test_brain_masks  = get_dataset(args.dataset_name,supervised=True)

# get dataloader
dataset = Data(X_train, Y_train, X_val, Y_val, X_test, Y_test, handler)
print(f"number of testing pool: {dataset.n_test}")
print()
# get network
net = get_net(args.dataset_name, device)

# start supervised learning baseline
dataset.supervised_training_labels()
labeled_idxs, labeled_data = dataset.get_labeled_data()
val_data = dataset.get_val_data()
print(f"number of labeled pool: {len(labeled_idxs)}")
net.supervised_val_loss(labeled_data,val_data,rd=0)
preds = net.predict(dataset.get_test_data(), full_test_imgs_list, x_test_slice, test_brain_images)
print(f"testing dice: {dataset.cal_test_acc(preds,test_brain_masks)}")