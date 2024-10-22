import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from time import time
from gnn import GNN_module

def np2cuda(array):
    tensor = torch.from_numpy(array)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name, map_location=lambda storage, loc: storage))
    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

###############################################################
## Vanilla CNN model, used to extract visual features

import torch
import torch.nn as nn

class EmbeddingCNN2D(nn.Module):

    def __init__(self, image_size, cnn_feature_size, cnn_hidden_dim, cnn_num_layers):
        super(EmbeddingCNN2D, self).__init__()

        module_list = []
        dim = cnn_hidden_dim
        for i in range(cnn_num_layers):
            if i == 0:
                module_list.append(nn.Conv2d(1, dim, 3, 1, 1, bias=False))
                module_list.append(nn.BatchNorm2d(dim))
            else:
                module_list.append(nn.Conv2d(dim, dim*2, 3, 1, 1, bias=False))
                module_list.append(nn.BatchNorm2d(dim*2))
                dim *= 2
            module_list.append(nn.MaxPool2d(2))
            module_list.append(nn.LeakyReLU(0.1, True))
            image_size //= 2
        module_list.append(nn.Conv2d(dim, cnn_feature_size, image_size, 1, bias=False))
        module_list.append(nn.BatchNorm2d(cnn_feature_size))
        module_list.append(nn.LeakyReLU(0.1, True))

        self.module_list = nn.ModuleList(module_list)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        for l in self.module_list:
            inputs = l(inputs)

        outputs = inputs.view(inputs.size(0), -1)
        return outputs

    def freeze_weight(self):
        for p in self.parameters():
            p.requires_grad = False

class EmbeddingCNN1D(nn.Module):

    def __init__(self, sequence_length, cnn_feature_size, cnn_hidden_dim, cnn_num_layers):
        super(EmbeddingCNN1D, self).__init__()

        module_list = []
        dim = cnn_hidden_dim
        for i in range(cnn_num_layers):
            if i == 0:
                module_list.append(nn.Conv1d(1, dim, 3, padding=1, bias=False))
                module_list.append(nn.BatchNorm1d(dim))
            else:
                module_list.append(nn.Conv1d(dim, dim*2, 3, padding=1, bias=False))
                module_list.append(nn.BatchNorm1d(dim*2))
                dim *= 2
            module_list.append(nn.MaxPool1d(2))
            module_list.append(nn.LeakyReLU(0.1, inplace=True))
            sequence_length //= 2
        module_list.append(nn.Conv1d(dim, cnn_feature_size, sequence_length, bias=False))
        module_list.append(nn.BatchNorm1d(cnn_feature_size))
        module_list.append(nn.LeakyReLU(0.1, inplace=True))

        self.module_list = nn.ModuleList(module_list)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        for l in self.module_list:
            inputs = l(inputs)

        outputs = inputs.view(inputs.size(0), -1)
        return outputs

    def freeze_weight(self):
        for p in self.parameters():
            p.requires_grad = False
    
class GNN(myModel):
    def __init__(self, cnn_feature_size, gnn_feature_size, nway):
        super(GNN, self).__init__()

        num_inputs = cnn_feature_size + nway
        graph_conv_layer = 2
        self.gnn_obj = GNN_module(nway=nway, input_dim=num_inputs,
            hidden_dim=gnn_feature_size, 
            num_layers=graph_conv_layer, 
            feature_type='dense')

    def forward(self, inputs):
        logits = self.gnn_obj(inputs).squeeze(-1)

        return logits
      
class gnnModel(myModel):
    def __init__(self, nway):
        super(myModel, self).__init__()
        sequence_length = 84
        cnn_feature_size = 64
        cnn_hidden_dim = 32
        cnn_num_layers = 3

        gnn_feature_size = 32
        image_size = 84
        self.cnn_feature = EmbeddingCNN2D(image_size, cnn_feature_size, cnn_hidden_dim, cnn_num_layers)
        # self.cnn_feature = EmbeddingCNN1D(sequence_length, cnn_feature_size, cnn_hidden_dim, cnn_num_layers)
        self.gnn = GNN(cnn_feature_size, gnn_feature_size, nway)

    def forward(self, data):
        [x, _, _, _, xi, _, one_hot_yi, _] = data

        z = self.cnn_feature(x)
        zi_s = [self.cnn_feature(xi[:, i, :]) for i in range(xi.size(1))]

        zi_s = torch.stack(zi_s, dim=1)


        # follow the paper, concatenate the information of labels to input features
        uniform_pad = torch.FloatTensor(one_hot_yi.size(0), 1, one_hot_yi.size(2)).fill_(
            1.0/one_hot_yi.size(2))
        uniform_pad = tensor2cuda(uniform_pad)

        labels = torch.cat([uniform_pad, one_hot_yi], dim=1)
        features = torch.cat([z.unsqueeze(1), zi_s], dim=1)

        nodes_features = torch.cat([features, labels], dim=2)

        out_logits = self.gnn(inputs=nodes_features)
        logsoft_prob = F.log_softmax(out_logits, dim=1)

        return logsoft_prob

class Trainer():
    def __init__(self, trainer_dict):

        self.outer_optimizer = None
        self.num_labels = 12

        self.args = trainer_dict['args']
        self.logger = trainer_dict['logger']

        if self.args.todo == 'train':
            self.tr_dataloader = trainer_dict['tr_dataloader']

        if self.args.model_type == 'gnn':
            Model = gnnModel
        
        self.model = Model(nway=self.args.nway)

        self.logger.info(self.model)

        self.total_iter = 0
        self.sample_size = 32
        
        self.adaptation_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.inner_lr)
                                                    # weight_decay=self.args.l2_reg_lambda)

    def maml_train(self):
        if self.args.freeze_cnn:
            self.model.cnn_feature.freeze_weight()
            print('freeze cnn weight...')

        best_loss = 1e8
        best_acc = 0.0
        stop = 0
        eval_sample = 5000
        self.model_cuda()
        self.model_dir = os.path.join(self.args.model_folder, 'model.pth')

        self.model.train()
        self.outer_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        start = time()
        tr_loss_list = []
        # 训练循环
        for i in range(self.args.max_iteration):
            # 1. 抽取任务
            support_data, query_data = self.tr_dataloader.load_tr_batch()
            # support_data_v, query_data_v = self.tr_dataloader.load_te_batch()
            init_model_state = self.model.state_dict()

            model_states = []
            # 2. 内循环：适应任务
            for j in range(3):
                self.adaptation_optimizer.zero_grad()
                input_data = [tensor[5*j:5*j+5] for tensor in support_data]
                _, adapted_model = self.adapt(input_data, init_model_state)
                model_states.append(adapted_model.state_dict())

            # 3. 外循环：评估和更新
            total_loss = torch.tensor(0.0).to(next(self.model.parameters()).device)  # 使用total_loss来累积损失
            for state in model_states:
                eval_model = self.model.load_state_dict(state)
                # 假设self.evaluate返回一个损失张量
                current_loss = self.evaluate(eval_model, query_data)
                total_loss += current_loss  # 累积损失
            # print(f"  Evaluation Loss: {current_loss.item()}")

            self.outer_optimizer.zero_grad()
            total_loss.requires_grad_(True)
            total_loss.backward()
            tr_loss_list.append(total_loss)
            self.outer_optimizer.step()

            if i % self.args.log_interval == 0:
                self.logger.info('iter: %d, spent: %.4f s, tr loss: %.5f' % (i, time() - start,
                    sum(tr_loss_list)/len(tr_loss_list)))

                del tr_loss_list[:]
                start = time()

            if i % self.args.eval_interval == 0:
                va_loss, va_acc = self.eval(self.tr_dataloader, eval_sample)

                self.logger.info('================== eval ==================')
                self.logger.info('iter: %d, va loss: %.5f, va acc: %.4f %%' % (i, va_loss, va_acc))
                self.logger.info('==========================================')

                if va_loss < best_loss:
                    stop = 0
                    best_loss = va_loss
                    best_acc = va_acc
                    if self.args.save:
                        self.model.save(self.model_dir)

                stop += 1
                start = time()

                if stop > self.args.early_stop:
                    break

            self.total_iter += 1

        self.logger.info('============= best result ===============')
        self.logger.info('best loss: %.5f, best acc: %.4f %%' % (best_loss, best_acc))

    def adapt(self, data, model_state):
        # 内循环适应过程

        adapted_model = self.model.__class__(self.args.nway)
        adapted_model.load_state_dict(model_state)
        adapted_model.to(next(self.model.parameters()).device)
        adaptation_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=0.2)
        adapted_model.train()
        for step in range(self.args.adaptation_steps):

            data_cuda = [tensor2cuda(_data) for _data in data]
            clip_value = 10.0
            logsoft_prob = adapted_model(data_cuda)
            loss = F.nll_loss(logsoft_prob, data_cuda[1])
            loss.backward()
            # print(f"Adaptation Step {step + 1}/{self.args.adaptation_steps}, Loss: {loss.item()}")

            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), clip_value)

            adaptation_optimizer.step()
            adaptation_optimizer.zero_grad()
        return data_cuda, adapted_model

    def evaluate(self, model, data):
        # 外循环评估过程
        self.model.eval()
        with torch.no_grad():
            logsoft_prob = self.model(data)
            loss = F.nll_loss(logsoft_prob, data[1])
        return loss
    
    def load_model(self, model_dir):
        self.model.load(model_dir)

        print('load model sucessfully...')

    def load_pretrain(self, model_dir):
        self.model.cnn_feature.load(model_dir)

        print('load pretrain feature sucessfully...')
    
    def model_cuda(self):
        if torch.cuda.is_available():
            self.model.cuda()

    def eval(self, dataloader, test_sample):
        model_copy = self.model.state_dict()

        args = self.args
        iteration = int(test_sample/self.args.batch_size)

        total_loss = 0.0
        total_sample = 0
        total_correct = 0
        for i in range(iteration):
            data = dataloader.load_te_batch(batch_size=args.batch_size,
                nway=args.nway, num_shots=args.shots)

            # 确保所有输入数据都在 GPU 上
            data_cuda = [tensor2cuda(_data) for _data in data]

            _, adapted_model = self.adapt(data, model_copy)
            adapted_model.eval()
            with torch.no_grad():
                logsoft_prob = adapted_model(data_cuda)
                pred = torch.argmax(logsoft_prob, dim=1)
                loss = F.nll_loss(logsoft_prob, data_cuda[1], reduction='sum')

                # Accumulate statistics
                total_loss += loss.item()
                total_sample += data_cuda[0].size(0)
                total_correct += torch.eq(pred, data_cuda[1]).float().sum().item()

            # Step 4: Restore the original model state
            self.model.load_state_dict(model_copy)

        accuracy = 100.0 * total_correct / total_sample
        return total_loss / total_sample, accuracy

    def train_batch(self):
        self.model.train()
        args = self.args

        data = self.tr_dataloader.load_tr_batch(batch_size=args.batch_size, 
            nway=args.nway, num_shots=args.shots)

        data_cuda = [tensor2cuda(_data) for _data in data]

        self.opt.zero_grad()

        logsoft_prob = self.model(data_cuda)

        # print('pred', torch.argmax(logsoft_prob, dim=1))
        # print('label', data[2])
        label = data_cuda[1]

        loss = F.nll_loss(logsoft_prob, label)
        loss.backward()
        self.opt.step()

        return loss.item()

    def train(self):
        if self.args.freeze_cnn:
            self.model.cnn_feature.freeze_weight()
            print('freeze cnn weight...')

        best_loss = 1e8
        best_acc = 0.0
        stop = 0
        eval_sample = 5000
        self.model_cuda()
        self.model_dir = os.path.join(self.args.model_folder, 'model.pth')

        self.opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr,
            weight_decay=1e-6)
        # self.opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, 
        #     weight_decay=1e-6)

        start = time()
        tr_loss_list = []
        for i in range(self.args.max_iteration):
            
            tr_loss = self.train_batch()
            tr_loss_list.append(tr_loss)

            if i % self.args.log_interval == 0:
                self.logger.info('iter: %d, spent: %.4f s, tr loss: %.5f' % (i, time() - start, 
                    np.mean(tr_loss_list)))
                del tr_loss_list[:]
                start = time()  

            if i % self.args.eval_interval == 0:
                va_loss, va_acc = self.eval(self.tr_dataloader, eval_sample)

                self.logger.info('================== eval ==================')
                self.logger.info('iter: %d, va loss: %.5f, va acc: %.4f %%' % (i, va_loss, va_acc))
                self.logger.info('==========================================')

                if va_loss < best_loss:
                    stop = 0
                    best_loss = va_loss
                    best_acc = va_acc
                    if self.args.save:
                        self.model.save(self.model_dir)

                stop += 1
                start = time()
            
                if stop > self.args.early_stop:
                    break

            self.total_iter += 1

        self.logger.info('============= best result ===============')
        self.logger.info('best loss: %.5f, best acc: %.4f %%' % (best_loss, best_acc))

    def test(self, test_data_array, te_dataloader):
        self.model_cuda()
        self.model.eval()
        start = 0
        end = 0
        args = self.args
        batch_size = args.batch_size
        pred_list = []

        with torch.no_grad():
            while start < test_data_array.shape[0]:
                end = start + batch_size 
                if end >= test_data_array.shape[0]:
                    batch_size = test_data_array.shape[0] - start

                data = te_dataloader.load_te_batch(batch_size=batch_size, nway=args.nway, 
                    num_shots=args.shots)

                test_x = test_data_array[start:end]

                data[0] = np2cuda(test_x)

                data_cuda = [tensor2cuda(_data) for _data in data]

                map_label2class = data[-1].cpu().numpy()

                logsoft_prob = self.model(data_cuda)
                # print(logsoft_prob)
                pred = torch.argmax(logsoft_prob, dim=1).cpu().numpy()

                pred = map_label2class[range(len(pred)), pred]

                pred_list.append(pred)

                start = end

        return np.hstack(pred_list)

    def pretrain_eval(self, loader, cnn_feature, classifier):
        total_loss = 0 
        total_sample = 0
        total_correct = 0

        with torch.no_grad():

            for j, (data, label) in enumerate(loader):
                data = tensor2cuda(data)
                label = tensor2cuda(label)
                output = classifier(cnn_feature(data))
                output = F.log_softmax(output, dim=1)
                loss = F.nll_loss(output, label)

                total_loss += loss.item() * output.shape[0]

                pred = torch.argmax(output, dim=1)

                assert pred.shape == label.shape

                total_correct += torch.eq(pred, label).float().sum().item()
                total_sample += pred.shape[0]

        return total_loss / total_sample, 100.0 * total_correct / total_sample

    def pretrain(self, pretrain_dataset, test_dataset):
        pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, 
                batch_size=self.args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                        batch_size=self.args.batch_size, shuffle=True)

        self.model_cuda()

        best_loss = 1e8
        self.model_dir = os.path.join(self.args.model_folder, 'pretrain_model.pth')

        cnn_feature = self.model.cnn_feature
        classifier = nn.Linear(list(cnn_feature.parameters())[-3].shape[0], self.num_labels)
        
        if torch.cuda.is_available():
            classifier.cuda()
        self.pretrain_opt =  torch.optim.Adam(
            list(cnn_feature.parameters()) + list(classifier.parameters()), 
            lr=self.args.lr, 
            weight_decay=1e-6)

        start = time()

        for i in range(10000):
            total_tr_loss = []
            for j, (data, label) in enumerate(pretrain_loader):
                data = tensor2cuda(data)
                label = tensor2cuda(label)
                output = classifier(cnn_feature(data))

                output = F.log_softmax(output, dim=1)
                loss = F.nll_loss(output, label)

                self.pretrain_opt.zero_grad()
                loss.backward()
                self.pretrain_opt.step()
                total_tr_loss.append(loss.item())

            te_loss, te_acc = self.pretrain_eval(test_loader, cnn_feature, classifier)
            self.logger.info('iter: %d, tr loss: %.5f, spent: %.4f s' % (i, np.mean(total_tr_loss), 
                time() - start))
            self.logger.info('--> eval: te loss: %.5f, te acc: %.4f %%' % (te_loss, te_acc))

            if te_loss < best_loss:
                stop = 0
                best_loss = te_loss
                if self.args.save:
                    cnn_feature.save(self.model_dir)

            stop += 1
            start = time()
        
            if stop > self.args.early_stop_pretrain:
                break



if __name__ == '__main__':
    import os
    b_s = 10
    nway = 5
    shots = 5
    batch_x = torch.rand(b_s, 1000).cuda()
    batches_xi = [torch.rand(b_s, 1000).cuda() for i in range(nway*shots)]

    label_x = torch.rand(b_s, nway).cuda()

    labels_yi = [torch.rand(b_s, nway).cuda() for i in range(nway*shots)]

    print('create model...')
    model = gnnModel(nway).cuda()
    # print(list(model.cnn_feature.parameters())[-3].shape)
    # print(len(list(model.parameters())))
    print(model([batch_x, label_x, None, None, batches_xi, None, labels_yi, None]).shape)
