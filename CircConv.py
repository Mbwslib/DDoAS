import torch
from torch import nn

class point_att(nn.Module):

    def __init__(self):
        super(point_att, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, snake_feature):

        proj_query = snake_feature.permute(0, 2, 1)
        energy = torch.bmm(proj_query, snake_feature)
        n_energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy = self.softmax(energy - n_energy)

        proj_value = snake_feature.permute(0, 2, 1)

        out = torch.bmm(energy, proj_value)
        out = out.permute(0, 2, 1)

        out = self.gamma*out + snake_feature
        return out

class DilatedCirConv(nn.Module):

    def __init__(self, state_dim, out_state_dim, n_adj=2, dilation=1):
        super(DilatedCirConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        self.circconv = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj * 2 + 1, dilation=self.dilation)

    def forward(self, x):
        x = torch.cat([x[..., -self.n_adj * self.dilation:], x, x[..., :self.n_adj * self.dilation]], dim=2)
        return self.circconv(x)


class BasicBlock(nn.Module):

    def __init__(self, state_dim, out_state_dim, n_adj=2, dilation=1):
        super(BasicBlock, self).__init__()

        self.circblock = DilatedCirConv(state_dim, out_state_dim, n_adj, dilation)
        self.circrelu = nn.ReLU(inplace=True)
        self.circnorm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.circblock(x)
        x = self.circrelu(x)
        x = self.circnorm(x)

        return x

class AttSnake(nn.Module):

    def __init__(self, n_adj):
        super(AttSnake, self).__init__()

        self.head = BasicBlock(130, 64, n_adj=n_adj)
        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        for i in range(self.res_layer_num):
            circconv = BasicBlock(64, 64, n_adj=n_adj, dilation=dilation[i])
            self.__setattr__('circconv'+str(i), circconv)

        self.fusion = nn.Conv1d(512, 128, 1)

        self.att_point = point_att()

    def forward(self, app_features):

        states = []

        x = self.head(app_features)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('circconv'+str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)
        snake_feature = self.att_point(self.fusion(state))
        #snake_feature = self.fusion(state)

        return snake_feature