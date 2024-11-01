import argparse
import numpy as np
import torch
import pandas as pd
from pprint import pprint

from data import Data
from utils import get_dataset, get_net, get_strategy
from config import parse_args
from seed import setup_seed

def main(param1,param2,param3):
    # args = parse_args.parse_known_args()[0]
    args = parse_args()
    pprint(vars(args))
    print()

    # fix random seed
    setup_seed(42)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test, handler, full_test_imgs_list, x_test_slice, test_brain_images, test_brain_masks = get_dataset(args.dataset_name,supervised=False)
    dataset = Data(X_train, Y_train, X_val, Y_val, X_test, Y_test, handler)

    net = get_net(args.dataset_name, device) # load network
    strategy = get_strategy(param1)(dataset, net) # load strategy

    # start experiment
    dataset.initialize_labels_random(args.n_init_labeled)
    print("Round 0")
    rd = 0
    strategy.train(rd, args.training_name)
    accuracy = []
    size = []
    preds= strategy.predict(dataset.get_test_data(), full_test_imgs_list, x_test_slice, test_brain_images) # get model prediction for test dataset
    print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds,test_brain_masks)}")  # get model performance for test dataset
    accuracy.append(dataset.cal_test_acc(preds,test_brain_masks))
    size.append(args.n_init_labeled)
    testing_accuracy = 0

    unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data(index = None)
    index = None
    print(f"number of labeled pool: {args.n_init_labeled}")
    print(f"number of unlabeled pool: {len(unlabeled_idxs)}")
    print(f"number of testing pool: {dataset.n_test}")
    print()
    # pseudo label筛选样本
    if param2 != None: 
        labels = net.predict_black_patch(unlabeled_data,param2)
        index = dataset.delete_black_patch(unlabeled_idxs, labels)
        unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data(index = index)
    print(f"number of labeled pool: {args.n_init_labeled}")
    print(f"number of unlabeled pool: {len(unlabeled_idxs)}")
    print(f"number of testing pool: {dataset.n_test}")
    print()


    for rd in range(1, args.n_round + 1):
        print(f"Round {rd}")
        # query
        if args.strategy_name == "AdversarialAttack":
            # query_idxs, generative_top_sample, real_end_sample, fake_label, pseudo_idxs = strategy.query(rd, args.n_query)
            query_idxs = strategy.query(args.n_query,index,param2,param3) #([500, 1, 128, 128])
            # label = dataset.get_label(query_idxs)
            # adversarial_sample = []
            # adversarial_label = []
            # pseudo_adversarial_sample = []
            # pseudo_label = []

            # query sample + adversarial sample
            # for one adversarial sample
            # for i in range(len(query_idxs)):
            # dataset.add_labeled_data(generative_top_sample, label)

        else:
            query_idxs = strategy.query(args.n_query,index,param2,param3)  # query_idxs为active learning请求标签的数据

        # update labels
        strategy.update(query_idxs)  # update training dataset and unlabeled dataset for active learning
        strategy.train(rd, args.training_name)

        # unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data(index = None)
        # labels = net.predict_black_patch(unlabeled_data)
        # index = dataset.delete_black_patch(unlabeled_idxs, labels)

        # efficient training
        # strategy.efficient_train(rd,dataset.get_train_data())

        # calculate accuracy
        preds= strategy.predict(dataset.get_test_data(), full_test_imgs_list, x_test_slice, test_brain_images)
        testing_accuracy = dataset.cal_test_acc(preds,test_brain_masks)
        print(f"Round {rd} testing accuracy: {testing_accuracy}")

        accuracy.append(testing_accuracy)
        labeled_idxs, _ = dataset.get_labeled_data()
        size.append(len(labeled_idxs))
        # unlabeled_idxs, _ = dataset.get_unlabeled_data(index = index)
        # if len(unlabeled_idxs) < 300:
        #     break

    # save the result
    dataframe = pd.DataFrame(
        {'model': 'Unet', 'Method': param1, 'Training dataset size': size, 'Accuracy': accuracy})
    dataframe.to_csv(f"./{param1}-42-{param2}-{param3}.csv", index=False, sep=',')

experiment_parameters = [
    {'param1': "AdversarialAttack", 'param2': 0.5, 'param3': 0.97},
    ]

for params in experiment_parameters:
    main(params['param1'],params['param2'],params['param3'])