import torch
from torch import nn
from torch.nn import functional as F


class Learner(nn.Module):
    def __init__(self, config):
        """
        param config: network config file, type:list of (string, list)
        """
        super(Learner, self).__init__()
        self.config = config

        self.net_vars = nn.ParameterList()  # this dict contains all tensors needed to be optimized
        self.net_vars_bn = nn.ParameterList()  # running_mean and running_var

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                w = nn.Parameter(torch.ones(*param[:4]))  # [ch_out, ch_in, kernel_size, kernel_size]
                torch.nn.init.kaiming_normal_(w)  # gain=1 according to cbfin's implementation
                self.net_vars.append(w)
                self.net_vars.append(nn.Parameter(torch.zeros(param[0])))  # [ch_out]

            elif name is 'complex_conv':
                w_real = nn.Parameter(torch.ones(*param[:3]))  # [ch_out, ch_in, kernel_size, kernel_size]
                torch.nn.init.kaiming_normal_(w_real)  # gain=1 according to cbfin's implementation
                self.net_vars.append(w_real)
                self.net_vars.append(nn.Parameter(torch.zeros(param[0])))  # [ch_out]
                w_img = nn.Parameter(torch.ones(*param[:3]))  # [ch_out, ch_in, kernel_size, kernel_size]
                torch.nn.init.kaiming_normal_(w_img)  # gain=1 according to cbfin's implementation
                self.net_vars.append(w_img)
                self.net_vars.append(nn.Parameter(torch.zeros(param[0])))  # [ch_out]

            elif name is 'conv_t2d':
                w = nn.Parameter(torch.ones(*param[:4]))  # [ch_in, ch_out, kernel_size, kernel_size, stride, padding]
                torch.nn.init.kaiming_normal_(w)  # gain=1 according to cbfin's implementation
                self.net_vars.append(w)
                self.net_vars.append(nn.Parameter(torch.zeros(param[1])))  # [ch_in, ch_out]

            elif name is 'linear':
                w = nn.Parameter(torch.ones(*param))  # [ch_out, ch_in]
                torch.nn.init.kaiming_normal_(w)  # gain=1 according to cbfinn's implementation
                self.net_vars.append(w)
                self.net_vars.append(nn.Parameter(torch.zeros(param[0])))  # [ch_out]

            elif name is 'bn':
                w = nn.Parameter(torch.ones(param[0]))  # [ch_out]
                self.net_vars.append(w)
                self.net_vars.append(nn.Parameter(torch.zeros(param[0])))  # [ch_out]

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.net_vars_bn.extend([running_mean, running_var])

            elif name in ['tanh', 'relu', 'up_sample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leaky_relu', 'sigmoid', 'max_pool1d']:
                continue

            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''
        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'complex_conv':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4],)
                info += tmp + '\n'

            elif name is 'conv_t2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name is 'leaky_relu':
                tmp = 'leaky_relu:(slope:%f)' % (param[0])
                info += tmp + '\n'

            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'

            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'

            elif name is 'max_pool1d':
                tmp = 'max_pool1d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'

            elif name in ['flatten', 'tanh', 'relu', 'up_sample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'

            else:
                raise NotImplementedError

        return info

    def forward(self, x, net_vars=None, bn_training=True):
        """
        This function can be called by fine_tuning, however, in fine_tuning, we do not wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weights.
        param x: [b, 1, 28, 28]
        param net_vars:
        param bn_training: set False to not update
        return: x, loss, likelihood, kld
        """

        if net_vars is None:
            net_vars = self.net_vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'conv2d':  # remember to keep synchronized of forward_encoder and forward_decoder!
                w, b = net_vars[idx], net_vars[idx+1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)

            elif name is 'complex_conv':  # remember to keep synchronized of forward_encoder and forward_decoder!
                w_re, b_re, w_img, b_img = net_vars[idx], net_vars[idx+1], net_vars[idx+2], net_vars[idx+3]
                x_real = x[:, 0:x.shape[1]//2, :]
                x_img = x[:, x.shape[1]//2: x.shape[1], :]
                real = F.conv1d(x_real, w_re, b_re, stride=param[3], padding=param[4]) \
                    - F.conv1d(x_img, w_img, b_img, stride=param[3], padding=param[4])
                imaginary = F.conv1d(x_img, w_re, b_re, stride=param[3], padding=param[4]) \
                    + F.conv1d(x_real, w_img, b_img, stride=param[3], padding=param[4])
                x = torch.cat((real, imaginary), dim=1)
                idx += 4
                # print(name, param, '\tout:', x.shape)

            elif name is 'conv_t2d':  # remember to keep synchronized of forward_encoder and forward_decoder!
                w, b = net_vars[idx], net_vars[idx+1]
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)

            elif name is 'linear':
                w, b = net_vars[idx], net_vars[idx+1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())

            elif name is 'bn':
                w, b = net_vars[idx], net_vars[idx+1]
                running_mean, running_var = self.net_vars_bn[bn_idx], self.net_vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                x = x.view(x.size(0), -1)
                embedding = x

            elif name is 'reshape':
                x = x.view(x.size(0), *param)

            elif name is 'relu':
                x = F.relu(x, inplace=param[0])

            elif name is 'leaky_relu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])

            elif name is 'tanh':
                x = F.tanh(x)

            elif name is 'sigmoid':
                x = torch.sigmoid(x)

            elif name is 'up_sample':
                x = F.upsample_nearest(x, scale_factor=param[0])

            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])

            elif name is 'max_pool1d':
                x = F.max_pool1d(x, param[0], param[1], param[2])

            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(net_vars)
        assert bn_idx == len(self.net_vars_bn)

        return embedding, x

    def zero_grad(self, net_vars=None):
        """
        param net_vars:
        return:
        """
        with torch.no_grad():
            if net_vars is None:
                for p in self.net_vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in net_vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self, *args, **kwargs):  # def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        return:
        """
        return self.net_vars
