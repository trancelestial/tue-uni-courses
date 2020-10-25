import torch
import torch.nn as nn
import torch.utils as utils
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
#from tqdm import tqdm
import torchvision
import pdb

skip = True

# default `log_dir` is "runs" - we'll be more specific here
saver_path = (f"/home/stud6/{'' if skip else 'no'}skip_log/")
writer = SummaryWriter(saver_path)

epoch = 20
batch_size = 300
learning_rate = 0.001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoisyFashionMNIST(Dataset):
    def __init__(self, incoming_df):

        super(NoisyFashionMNIST, self).__init__()
        self.incoming_df = incoming_df


    def __len__(self):
        return len(self.incoming_df)

    def __getitem__(self, idx):

        # TODO: add noise

        image, target = self.incoming_df[idx]

        noise = torch.rand_like(image)

        noisy_image = noise + image

        return noisy_image, image


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        #TODO : model
        self.conv1_1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv1_3 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.trans_conv1_1 = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1)
        self.trans_conv1_2 = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1)
        self.conv2_1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.out = nn.Conv2d(16, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))

        x = F.relu(self.trans_conv1_1(x))
        x = F.relu(self.trans_conv1_2(x))
        x = self.conv2_1(x)

        return self.out(x)
        # return F.sigmoid(self.out(x))

# this is the model has ship connections as in U-Net (concat)
class ModelSkipU(nn.Module):

    def __init__(self):
        super(ModelSkip, self).__init__()

        #TODO : model
        self.conv1_1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv1_3 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.trans_conv1_1 = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1)
        self.trans_conv1_2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.conv2_1 = nn.Conv2d(17, 16, 3, stride=1, padding=1)
        self.out = nn.Conv2d(16, 1, 3, stride=1, padding=1)
    
    def _skip(self, _from, _to):
        return torch.cat([_from, _to], axis=1)

    def forward(self, x):
        x0 = x
        x1 = x = F.relu(self.conv1_1(x))
        lat = x = F.relu(self.conv1_2(x))

        x = F.relu(self.trans_conv1_1(x))
        x = F.relu(self.trans_conv1_2(self._skip(x1,x)))
        x = F.relu(self.conv2_1(self._skip(x0,x)))

        return self.out(x)
        # return F.sigmoid(self.out(x))

# this is the model has ship connections as in ResNet (add) - one in exercise
class ModelSkip(nn.Module):

    def __init__(self):
        super(ModelSkip, self).__init__()

        #TODO : model
        self.conv1_1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv1_3 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.trans_conv1_1 = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1)
        self.trans_conv1_2 = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1)
        self.conv2_1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.out = nn.Conv2d(16, 1, 3, stride=1, padding=1)
    
    def forward(self, x):
        x0 = x
        x1 = x = F.relu(self.conv1_1(x))
        lat = x = F.relu(self.conv1_2(x))

        x = F.relu(self.trans_conv1_1(x))
        x = F.relu(self.trans_conv1_2(x1+x))
        x = F.relu(self.conv2_1(x))

        return self.out(x)
 
model = (ModelSkip() if skip else Model()).to(device=device)
# Data

fmnist_train = dset.FashionMNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None,
                                 download=True)
fmnist_test = dset.FashionMNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None,
                                download=True)
# noise
fmnist_noisy_train = NoisyFashionMNIST(fmnist_train)
fmnist_noisy_test = NoisyFashionMNIST(fmnist_test)


# TODO : Batch and shuffle
# Batch and prepare
train_loader = DataLoader(fmnist_noisy_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(fmnist_noisy_test, batch_size=batch_size, shuffle=True)

# Check output of autoencoder
test_in = fmnist_train[0][0].unsqueeze(0).to(device)
test_out = model(test_in)
print(f'Input shape: {test_in.shape}\nOutput shape: {test_out.shape}')
assert(test_in.shape == test_out.shape)

# Set some more stuff
parameters = list(model.parameters())

#TODO: set optimizer
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

for i in range(epoch):


    loss = 0.
    loss_test = 0.

    # TRAIN

    for idx, batch_info in enumerate(train_loader):

        model.train()

        #TODO: fill data
        noisy_image = batch_info[0]
        clean_image = batch_info[1]

        image = clean_image.to(device=device)
        image_n = noisy_image.to(device=device)

        output = model(image_n)

        loss = torch.abs(output - image).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('training loss',
                          loss.item(),
                          i * len(train_loader) + idx)

        images = torch.cat([image_n.cpu().repeat(1, 3, 1, 1), output.cpu().repeat(1, 3, 1, 1), image.cpu().repeat(1, 3, 1, 1)], axis=3)
        img_grid = torchvision.utils.make_grid(images.cpu())
        # pdb.set_trace()
        writer.add_image('Input VS Output VS Ground Truth TRAINING',
                         img_grid,
                         global_step=i * len(train_loader) + idx)

    # TEST

    for idx_test, batch_info_test in enumerate(test_loader):

        optimizer.zero_grad()

        with torch.no_grad():

            # TODO: fill test data

            noisy_image_t = batch_info_test[0]
            clean_image_t = batch_info_test[1]

            image_t = clean_image_t.to(device=device)
            image_n_t = noisy_image_t.to(device=device)

            output_t = model(image_n_t)

            loss_test = torch.abs(output_t - image_t).mean()

            writer.add_scalar('Validation loss',
                              loss_test.item(),
                              i * len(test_loader) + idx_test)

            images_test = torch.cat([image_n_t.cpu().repeat(1, 3, 1, 1), output_t.cpu().repeat(1, 3, 1, 1),image_t.cpu().repeat(1, 3, 1, 1)], axis=3)
            img_grid_t = torchvision.utils.make_grid(images_test.cpu())
            # pdb.set_trace()
            writer.add_image('Input VS Output VS Ground Truth VALIDATION',
                             img_grid_t,
                             global_step=i * len(test_loader) + idx_test)

    print('epoch done: ', i)
    # torch.save([model], PATH)
    print('Train Loss :', loss.item())
    print('Validation loss : ', loss_test.item())
