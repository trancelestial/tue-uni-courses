import torch
import random
import time

import numpy as np
import collections

import logging
from PIL import Image, ImageOps

#from src.imitations import load_imitations

def train(data_folder: str, trained_network_file: str, net, optimizer_name='adam', lr=1e-3, n_epochs=100,
          lr_milestones: tuple = (), batch_size=64, weight_decay = 1e-6,
          augment_data: bool=True, balanced: bool=False, shuffle: bool=False, save_losses: bool=False,
          device='cuda', timestamp=False):
    """
    Function for training the network.
    """
    logger = logging.getLogger()

    net = net.to(device)
    # net = ClassificationNetwork(number_of_classes).to(device)
    # net = net.double()
    observations, actions = load_imitations(data_folder)
    logger.info(f'Data loaded.')

    augmented_observations = []
    augmented_actions = []

    # augmenting the data by flipping images of turning
    if augment_data:
        for i in range(len(actions)):
            if actions[i][0] != 0:  # Flipping only images of turning
                # print(observations[i].shape)
                im = ImageOps.mirror(Image.fromarray(observations[i][:84,:,:], 'RGB'))
                ap = np.copy(observations[i])
                ap[:84,:,:] = np.array(im)
                # Image.fromarray(ap, 'RGB').show()
                # Image.fromarray(observations[i], 'RGB').show()
                augmented_actions.append(np.copy(actions[i]) * [-1, 1, 1])
                augmented_observations.append(ap)

        augmented_observations = np.array(augmented_observations)
        augmented_actions = np.array(augmented_actions)
        # print(augmented_actions.shape)
        # print(augmented_observations.shape)

        observations = np.array(observations)
        actions = np.array(actions)
        # print(f'Shapes before augment: \n{observations.shape}\n{actions.shape}')
        observations = np.concatenate((observations, augmented_observations), axis=0)
        actions = np.concatenate((actions, augmented_actions), axis=0)
        print(f'Shapes after augment: \n{observations.shape}\n{actions.shape}')

        logger.info(f'Data augmented.')

    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    # shuffling data
    if shuffle:
        logger.info(f'  Shuffling data...')
        zipped = list(zip(observations, actions))
        random.shuffle(zipped)
        observations, actions = zip(*zipped)
        logger.info(f'   Done.')

    class_balance_tensor = torch.ones(net.number_of_classes).to(device)

    # handle imbalance by calculating weights
    if balanced:
        diff_count = collections.Counter(np.array(net.classes_to_labels(net.actions_to_classes(actions))))
        diff_count = collections.OrderedDict(sorted(diff_count.items(), key=lambda t: t[0]))
        n_classes_train = len(diff_count)

        logger.info(f'Count of classes: {diff_count}\n')

        if n_classes_train != net.number_of_classes:
            print(f'Classes in train: {n_classes_train} != {net.number_of_classes}')
            exit(1)
        n_max = max(diff_count.values())
        weights = [n_max / n_class for n_class in diff_count.values()]
        class_balance_tensor = torch.Tensor(weights).to(device)
        logger.info(f'Weights: {weights}')

    batches = [batch for batch in zip(observations, net.actions_to_classes(actions))]

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay,
                                 amsgrad=optimizer_name == 'amsgrad')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones, gamma=0.1)

    losses_array = []

    logging.info(f'Starting training...')
    start_time = time.time()
    net.train()
    for epoch in range(n_epochs):
        random.shuffle(batches)

        scheduler.step()
        if epoch in lr_milestones:
            logger.info(f'   LR scheduler: new learning rate is {scheduler.get_lr()[0]}')

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0), (-1, 96, 96, 3))

                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0), (-1, net.number_of_classes))

                batch_out = net(batch_in)

                optimizer.zero_grad()

                if net.type == 'regressor':
                    loss = mean_squared_error(batch_out, batch_gt)
                elif net.type == 'classifier':
                    loss = cross_entropy_loss(batch_out, batch_gt, class_balance_tensor)
                else:
                    print('Some weird net we have here, don\'t we?!')
                    exit(1)

                loss.backward()
                optimizer.step()

                total_loss += loss

                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (n_epochs - 1 - epoch)
        losses_array.append(total_loss.item())
        logger.info(f'  Epoch {epoch+1:4d}\tloss: {total_loss:.6f}\tETA: +{time_left:.1f}')

    logger.info(f'Training time {time.time() - start_time:.3f}')
    logger.info(f'Finished training.')

    # adding timestamp to the file name
    if timestamp:
        dot_pos = trained_network_file.rindex('.')
        trained_network_file = trained_network_file[:dot_pos] + '_' + time.strftime('%Y-%m-%d-%H:%M:%S') \
                               + '_' + str(net.number_of_classes) + 'class' + trained_network_file[dot_pos:]

    # saving network
    torch.save(net, trained_network_file)
    logger.info(f'Network {trained_network_file[trained_network_file.rindex("/")+1:]} saved.')

    # saving losses for plotting
    if save_losses:
        trained_network_name = '../losses/training_loss_' + time.strftime('%Y-%m-%d-%H:%M:%S') \
                               + '_' + str(net.number_of_classes) + 'class'
        # file = os.open(trained_network_file, 'wb')
        # pickle.dump(np.array(losses_array), trained_network_name)
        np.save(trained_network_name, losses_array)
        logger.info(f'Losses saved.')

def mean_squared_error(batch_out, batch_gt):
    loss = (batch_gt - batch_out) ** 2

    return torch.mean(torch.sum(loss, dim=1), dim=0)

def cross_entropy_loss(batch_out, batch_gt, class_balance_tensor):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    batch_out:      torch.Tensor of size (batch_size, number_of_classes)
    batch_gt:       torch.Tensor of size (batch_size, number_of_classes)
    return          float
    """

    eps = 1e-6
    loss = class_balance_tensor * (batch_gt * torch.log(batch_out + eps) + (1 - batch_gt) * torch.log(1 - batch_out + eps))

    return -torch.mean(torch.sum(loss, dim=1), dim=0)
