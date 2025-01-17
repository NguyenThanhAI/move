import argparse
import json

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.move_model import MOVEModel
from models.move_model_nt import MOVEModelNT


class MOVEDatasetFull(Dataset):
    """
    MOVEDataset object returns one song from the test data.
    Given features are in their full length.
    """

    def __init__(self, data):
        """
        Initialization of the MOVEDataset object
        :param data: pcp features
        :param labels: labels of features (should be in the same order as features)
        :param h: height of pcp features (number of bins, e.g. 12 or 23)
        :param w: width of pcp features (number of frames in the temporal dimension)
        """
        self.data = data  # pcp features

    def __getitem__(self, index):
        """
        getitem function for the MOVEDataset object
        :param index: index of the song picked by the dataloader
        :return: pcp feature of the selected song
        """

        item = self.data[index]

        return item.float()

    def __len__(self):
        """
        Size of the MOVEDataset object
        :return: length of the entire dataset
        """
        return len(self.data)


def evaluate(save_name,
             model_type,
             emb_size,
             sum_method,
             final_activation):
    print("Prepare random dataset")
    num_examples = 1000
    test_data = []
    for i in tqdm(range(num_examples)):
        t_length = np.random.randint(1000, 2000)
        cremaPCP = np.random.rand(t_length, 12)
        cremaPCP_tensor = torch.from_numpy(cremaPCP).t()
        cremaPCP_reshaped = torch.cat((cremaPCP_tensor, cremaPCP_tensor))[:23].unsqueeze(0)
        test_data.append(cremaPCP_reshaped)

    test_map_set = MOVEDatasetFull(data=test_data)
    test_map_loader = DataLoader(test_map_set, batch_size=1, shuffle=False)

    print("Initialize model")
    # initializing the model
    if model_type == 0:
        move_model = MOVEModel(emb_size=emb_size, sum_method=sum_method, final_activation=final_activation)
    elif model_type == 1:
        move_model = MOVEModelNT(emb_size=emb_size, sum_method=sum_method, final_activation=final_activation)

    # loading a pre-trained model
    model_name = 'saved_models/model_{}.pt'.format(save_name)

    print("Load model")
    move_model.load_state_dict(torch.load(model_name, map_location='cpu'))
    move_model.eval()

    # sending the model to gpu, if available
    if torch.cuda.is_available():
        move_model.cuda()

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print("Run extract feature")
    with torch.no_grad():  # deactivating gradient tracking for testing
        move_model.eval()  # setting the model to evaluation mode

        # tensor for storing all the embeddings obtained from the test set
        embed_all = torch.tensor([], device=device)

        for batch_idx, item in tqdm(enumerate(test_map_loader)):

            if torch.cuda.is_available():  # sending the pcp features and the labels to cuda if available
                item = item.cuda()

            res_1 = move_model(item)  # obtaining the embeddings of each song in the mini-batch

            embed_all = torch.cat((embed_all, res_1))  # adding the embedding of the current song to the others

    return embed_all.cpu()


if __name__:
    with open('data/move_defaults.json') as f:
        defaults = json.load(f)

    parser = argparse.ArgumentParser(description='Training code of MOVE')
    parser.add_argument('-rt',
                        '--run_type',
                        type=str,
                        default='train',
                        choices=('train', 'test'),
                        help='Whether to run train or test script')
    parser.add_argument('-tp',
                        '--train_path',
                        type=str,
                        default=defaults['train_path'],
                        help='Path for training data. If more than one file are used, '
                             'write only the common part')
    parser.add_argument('-ch',
                        '--chunks',
                        type=int,
                        default=defaults['chunks'],
                        help='Number of chunks for training set')
    parser.add_argument('-vp',
                        '--val_path',
                        type=str,
                        default=defaults['val_path'],
                        help='Path for validation data')
    parser.add_argument('-sm',
                        '--save_model',
                        type=int,
                        default=defaults['save_model'],
                        choices=(0, 1),
                        help='1 for saving the trained model, 0 for otherwise')
    parser.add_argument('-ss',
                        '--save_summary',
                        type=int,
                        default=defaults['save_summary'],
                        choices=(0, 1),
                        help='1 for saving the training log, 0 for otherwise')
    parser.add_argument('-rs',
                        '--random_seed',
                        type=int,
                        default=defaults['random_seed'],
                        help='Random seed')
    parser.add_argument('-noe',
                        '--num_of_epochs',
                        type=int,
                        default=defaults['num_of_epochs'],
                        help='Number of epochs for training')
    parser.add_argument('-m',
                        '--model_type',
                        type=int,
                        default=defaults['model_type'],
                        choices=(0, 1),
                        help='0 for MOVE, 1 for MOVE without pitch transposition')
    parser.add_argument('-emb',
                        '--emb_size',
                        type=int,
                        default=defaults['emb_size'],
                        help='Size of the final embeddings')
    parser.add_argument('-sum',
                        '--sum_method',
                        type=int,
                        choices=(0, 1, 2, 3, 4),
                        default=defaults['sum_method'],
                        help='0 for max-pool, 1 for mean-pool, 2 for autopool, '
                             '3 for multi-channel attention, 4 for multi-channel adaptive attention')
    parser.add_argument('-fa',
                        '--final_activation',
                        type=int,
                        choices=(0, 1, 2, 3),
                        default=defaults['final_activation'],
                        help='0 for no activation, 1 for sigmoid, 2 for tanh, 3 for batch norm')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=defaults['learning_rate'],
                        help='Initial learning rate')
    parser.add_argument('-lrs',
                        '--lr_schedule',
                        type=int,
                        default=defaults['lr_schedule'],
                        choices=(0, 1, 2),
                        help='0 for no lr_schedule, 1 for decreasing lr at epoch 80, '
                             '2 for decreasing lr at epochs [80, 100]')
    parser.add_argument('-lrsf',
                        '--lrsch_factor',
                        type=float,
                        default=defaults['lrsch_factor'],
                        help='Factor for lr scheduler')
    parser.add_argument('-mo',
                        '--momentum',
                        type=float,
                        default=defaults['momentum'],
                        help='Value for momentum parameter for SGD')
    parser.add_argument('-pl',
                        '--patch_len',
                        type=int,
                        default=defaults['patch_len'],
                        help='Size of the input len in time dimension')
    parser.add_argument('-nol',
                        '--num_of_labels',
                        type=int,
                        default=defaults['num_of_labels'],
                        help='Number of cliques per batch for triplet mining')
    parser.add_argument('-da',
                        '--data_aug',
                        type=int,
                        choices=(0, 1),
                        default=defaults['data_aug'],
                        help='0 for no data aug, 1 using it')
    parser.add_argument('-nd',
                        '--norm_dist',
                        type=int,
                        choices=(0, 1),
                        default=defaults['norm_dist'],
                        help='1 for normalizing the distance, 0 for avoiding it')
    parser.add_argument('-ms',
                        '--mining_strategy',
                        type=int,
                        default=defaults['mining_strategy'],
                        choices=(0, 1, 2),
                        help='0 for only random, 1 for only semi-hard, 2 for only hard')
    parser.add_argument('-ma',
                        '--margin',
                        type=float,
                        default=defaults['margin'],
                        help='Margin for triplet loss')
    parser.add_argument('-ytc',
                        '--ytc_labels',
                        type=int,
                        default=defaults['ytc_labels'],
                        choices=(0, 1),
                        help='0 for using full training data, 1 for removing overlapping labels with ytc')
    parser.add_argument('-d',
                        '--dataset',
                        type=int,
                        choices=(0, 1, 2),
                        default=0,
                        help='Choosing evaluation set for testing. 0 for move validation, '
                             '1 for test on da-tacos, 2 for test on ytc')
    parser.add_argument('-dn',
                        '--dataset_name',
                        type=str,
                        default='',
                        help='Specifying a dataset name for evaluation. '
                             'The dataset must be located in the data folder')

    args = parser.parse_args()

    lr_arg = '{}'.format(args.learning_rate).replace('.', '-')
    margin_arg = '{}'.format(args.margin).replace('.', '-')

    save_name = 'move'

    for key in defaults.keys():
        if key == 'abbr':
            pass
        else:
            if defaults[key] != getattr(args, key):
                save_name = '{}_{}_{}'.format(save_name, defaults['abbr'][key], getattr(args, key))

    print("Save name: {}".format(save_name))

    embed_all = evaluate(save_name=save_name,
                         model_type=args.model_type,
                         emb_size=args.emb_size,
                         sum_method=args.sum_method,
                         final_activation=args.final_activation)

    embed_all = embed_all.numpy()
    print("Embedding shape: {}".format(embed_all.shape))
