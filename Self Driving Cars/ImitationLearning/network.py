import torch
import numpy as np
import logging

class ClassificationNetwork(torch.nn.Module):
    def __init__(self, number_of_classes: int):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()

        self.number_of_classes=number_of_classes
        self.logger = logging.getLogger(self.__class__.__name__)
        # added for the sake of keeping the same train function (not the prettiest thing but for the next one we will
        # use abstractions :) )
        self.type = 'classifier'

        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        # cropped image + 1 for speed
        # 2 maxpool
        # self.fc1 = torch.nn.Linear(32*9*10+1+4+1+1, 64)
        self.fc1 = torch.nn.Linear(32*84*96+1+4+1+1, 64)
        self.bnfc1 = torch.nn.BatchNorm1d(64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.bnfc2 = torch.nn.BatchNorm1d(32)
        self.fc3 = torch.nn.Linear(32, self.number_of_classes)


    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        batch_size = observation.shape[0]

        speed, abs, steer, gyro = self.extract_sensor_values(observation, observation.shape[0])

        observation = observation[:, :84, :, :].permute(0, 3, 1, 2)
        observation = observation / 255.0
        # observation = observation[:, 0, :, :] * 0.2989 + \
        #               observation[:, 1, :, :] * 0.5870 + \
        #               observation[:, 2, :, :] * 0.1140
        # observation = observation.unsqueeze(1)

        x = self.conv1(observation)
        # x = torch.nn.MaxPool2d(3)(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bn1(x)

        x = self.conv2(x)
        # x = torch.nn.MaxPool2d(3)(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bn2(x)

        x = self.conv3(x)
        # x = torch.nn.MaxPool2d(3)(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bn3(x)

        x = self.fc1(torch.cat([x.reshape(batch_size, -1), speed, abs, steer, gyro], dim=1))
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bnfc1(x)

        x = self.fc2(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bnfc2(x)

        x = self.fc3(x)

        out = torch.nn.Softmax(dim=1)(x)

        return out

    def classes_to_labels(self, classes):
        """
        Converts one hot encoded classes to integer labels
        :param classes: python list of N torch.Tensors of size number_of_classes
        :return:        python list of N integers
        """
        return [torch.argmax(oh_class) for oh_class in classes]

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """
        if self.number_of_classes == 9:
            oh_classes = [torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0]), # nothing
                          torch.Tensor([0, 1, 0, 0, 0, 0, 0, 0, 0]), # only left
                          torch.Tensor([0, 0, 1, 0, 0, 0, 0, 0, 0]), # only right
                          torch.Tensor([0, 0, 0, 1, 0, 0, 0, 0, 0]), # brake
                          torch.Tensor([0, 0, 0, 0, 1, 0, 0, 0, 0]), # gas
                          torch.Tensor([0, 0, 0, 0, 0, 1, 0, 0, 0]), # brake left
                          torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0, 0]), # brake right
                          torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1, 0]), # gas left
                          torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 1])  # gas right
                          ]
            # Ugly change dict
            classes = [oh_classes[0] if torch.all(torch.eq(action, torch.Tensor([0.0, 0.0, 0.0]))) else
                       (
                           oh_classes[5] if torch.all(torch.eq(action, torch.Tensor([-1.0, 0.0, 0.8]))) else (
                               oh_classes[6] if torch.all(torch.eq(action, torch.Tensor([1.0, 0.0, 0.8]))) else (
                                   oh_classes[7] if torch.all(torch.eq(action, torch.Tensor([-1.0, 0.5, 0.0]))) else (
                                       oh_classes[8] if torch.all(torch.eq(action, torch.Tensor([1.0, 0.5, 0.0]))) else (
                                           oh_classes[1] if action[0] == -1.0 else (
                                               oh_classes[2] if action[0] == 1.0 else (
                                                   oh_classes[3] if torch.all(
                                                       torch.eq(action, torch.Tensor([0.0, 0.0, 0.8]))) else (
                                                       oh_classes[4] if torch.all(
                                                           torch.eq(action, torch.Tensor([0.0, 0.5, 0.0]))) else (
                                                           print(f'Wuuut?!\n\t{action}')
                                                       )
                                                   )
                                               )
                                           )
                                       )
                                   )
                               )
                           )
                       ) for action in actions]

        if self.number_of_classes == 7:
            oh_classes = [torch.Tensor([1, 0, 0, 0, 0, 0, 0]),  # nothing
                          torch.Tensor([0, 1, 0, 0, 0, 0, 0]),  # only left
                          torch.Tensor([0, 0, 1, 0, 0, 0, 0]),  # only right
                          torch.Tensor([0, 0, 0, 1, 0, 0, 0]),  # brake
                          torch.Tensor([0, 0, 0, 0, 1, 0, 0]),  # gas
                          torch.Tensor([0, 0, 0, 0, 0, 1, 0]),  # brake left
                          torch.Tensor([0, 0, 0, 0, 0, 0, 1]),  # brake rightt
                          ]

            classes = [oh_classes[0] if torch.all(torch.eq(action, torch.Tensor([0.0, 0.0, 0.0]))) else
                            (
                                oh_classes[5] if torch.all(torch.eq(action, torch.Tensor([-1.0, 0.0, 0.8]))) else(
                                    oh_classes[6] if torch.all(torch.eq(action, torch.Tensor([1.0, 0.0, 0.8]))) else (
                                        oh_classes[1] if action[0] == -1.0 else (
                                            oh_classes[2] if action[0] == 1.0 else (
                                                oh_classes[3] if torch.all(torch.eq(action, torch.Tensor([0.0, 0.0, 0.8]))) else (
                                                    oh_classes[4] if torch.all(torch.eq(action, torch.Tensor([0.0, 0.5, 0.0]))) else (
                                                        print(f'Wuut?!\n\t{action}')
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            ) for action in actions]

        return classes

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        if self.number_of_classes == 9:
            action_switch = [[0.0, 0.0, 0.0],
                             [-1.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.8],
                             [0.0, 0.5, 0.0],
                             [-1.0, 0.0, 0.8],
                             [1.0, 0.0, 0.8],
                             [-1.0, 0.5, 0.0],
                             [1.0, 0.5, 0.0]]

        if self.number_of_classes == 7:
            action_switch = [[0.0, 0.0, 0.0],
                             [-1.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.8],
                             [0.0, 0.5, 0.0],
                             [-1.0, 0.0, 0.8],
                             [1.0, 0.0, 0.8]]

        action = action_switch[torch.argmax(scores)]

        return action

    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope

    def summary(self):
        net_parameters = filter(lambda p: p.requres_grad, self.parameters())
        print(f'[INFO] net_parameters: {net_parameters}')
        params = sum([np.prod(p.size())] for p in net_parameters)
        self.logger.info(f'Trainable parameters: {params}')
        self.logger.info(self)

# 1 b)
class MulticlassClassificationNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.number_of_classes = 4
        self.type = 'classifier'

        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)

        self.fc1 = torch.nn.Linear(32 * 84 * 96 + 1 + 4 + 1 + 1, 64)
        self.bnfc1 = torch.nn.BatchNorm1d(64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.bnfc2 = torch.nn.BatchNorm1d(32)
        self.fc3 = torch.nn.Linear(32, self.number_of_classes)


    def forward(self, observation):
        batch_size = observation.shape[0]

        speed, abs, steer, gyro = self.extract_sensor_values(observation, observation.shape[0])

        observation = observation[:, :84, :, :].permute(0, 3, 1, 2)
        observation = observation / 255.0

        x = self.conv1(observation)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bn3(x)

        x = self.fc1(torch.cat([x.reshape(batch_size, -1), speed, abs, steer, gyro], dim=1))
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bnfc1(x)

        x = self.fc2(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bnfc2(x)

        x = self.fc3(x)
        out = torch.nn.Sigmoid()(x)

        return out

    def actions_to_classes(self, actions):
        return [torch.Tensor([int(action[0] < 0.), int(action[0] > 0.), int(action[1] > 0.), int(action[2] > 0.)])
                for action in actions]

    def scores_to_action(self, scores):
        steer_difference = scores[0][0] - scores[0][1]
        steer = -1.0 if steer_difference > 0.5 else 0.0
        steer = 1.0 if steer_difference < -0.5 else steer
        gas = 0.5 if scores[0][2] > 0.5 else 0.0
        brake = 0.8 if scores[0][3] > 0.5 else 0.0

        return steer, gas, brake

    def classes_to_labels(self, classes):
        """
        Converts one hot encoded classes to integer labels
        :param classes: python list of N torch.Tensors of size number_of_classes
        :return:        python list of N integers
        """
        return [torch.argmax(oh_class) for oh_class in classes]


    def extract_sensor_values(self, observation, batch_size):
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope


    def summary(self):
        net_parameters = filter(lambda p: p.requres_grad, self.parameters())
        print(f'[INFO] net_parameters: {net_parameters}')
        params = sum([np.prod(p.size())] for p in net_parameters)
        self.logger.info(f'Trainable parameters: {params}')
        self.logger.info(self)

# 1 c)
class RegressionNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        # in this case it's number of outputs, name just for the sake of compatibility
        self.number_of_classes = 3
        self.type = 'regressor'

        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)

        self.fc1 = torch.nn.Linear(32 * 84 * 96 + 1 + 4 + 1 + 1, 64)
        self.bnfc1 = torch.nn.BatchNorm1d(64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.bnfc2 = torch.nn.BatchNorm1d(32)
        self.fc3 = torch.nn.Linear(32, self.number_of_classes)


    def forward(self, observation):
        batch_size = observation.shape[0]

        speed, abs, steer, gyro = self.extract_sensor_values(observation, observation.shape[0])

        observation = observation[:, :84, :, :].permute(0, 3, 1, 2)
        observation = observation / 255.0

        x = self.conv1(observation)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bn3(x)

        x = self.fc1(torch.cat([x.reshape(batch_size, -1), speed, abs, steer, gyro], dim=1))
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bnfc1(x)

        x = self.fc2(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.bnfc2(x)

        out = self.fc3(x)

        return out

    # Name is the same just for compatibility with training function used so far.
    # There are no classes in regression net.
    def actions_to_classes(self, actions):
        return actions

    # Same argument for the name. No smart logic here, just clip
    def scores_to_action(self, scores):
        return scores[0][0].item(), scores[0][1].item(), scores[0][2].item()


    def extract_sensor_values(self, observation, batch_size):
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope


    def summary(self):
        net_parameters = filter(lambda p: p.requres_grad, self.parameters())
        print(f'[INFO] net_parameters: {net_parameters}')
        params = sum([np.prod(p.size())] for p in net_parameters)
        self.logger.info(f'Trainable parameters: {params}')
        self.logger.info(self)
