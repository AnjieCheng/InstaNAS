from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import numpy as np
from torch.autograd import Variable
from utils import AverageMeter, adjust_learning_rate, error
import time
from tqdm import tqdm
class Trainer(object):
    def __init__(self, model, criterion=None, optimizer=None, args=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args

    def train(self, train_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()
        lr = adjust_learning_rate(self.optimizer, self.args.lr,
                                  self.args.decay_rate, epoch,
                                  self.args.epochs)

        if epoch > self.args.epochs // 2:
            drop_rate = 1-(0.5/(self.args.epochs//2))*(epoch-self.args.epochs//2)
        else:
            drop_rate = 1
        print('Epoch {:3d} lr = {:.6e} dr = {:.6e}'.format(epoch, lr, drop_rate))

        end = time.time()
        for i, (input, target) in tqdm(enumerate(train_loader)):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = Variable(input)
            target_var = Variable(target)

            '''
            Define Drop Rate and policy
            '''
            """
            policy_shape = (input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)

            if epoch > self.args.epochs // 2:
                policy = Variable(torch.from_numpy(np.random.binomial(1, drop_rate, policy_shape))).long().cuda()
            else:
                policy = Variable(torch.ones(input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)).long().cuda()
            """
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure error and record loss
            err1, err5 = error(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(err1.item(), input.size(0))
            top5.update(err5.item(), input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.args.print_freq > 0 and \
                    (i + 1) % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Err@1 {top1.val:.4f}\t'
                      'Err@5 {top5.val:.4f}'.format(
                          epoch, i + 1, len(train_loader),
                          batch_time=batch_time, data_time=data_time,
                          loss=losses, top1=top1, top5=top5))
        print('Epoch: {:3d} Train loss {loss.avg:.4f} '
              'Err@1 {top1.avg:.4f}'
              ' Err@5 {top5.avg:.4f}'
              .format(epoch, loss=losses, top1=top1, top5=top5))
        return losses.avg, top1.avg, top5.avg, lr, 0

    def test(self, val_loader, epoch, silence=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        end = time.time()
        total_start = time.time()
        print(len(val_loader))
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = Variable(input, volatile=True)
            target_var = Variable(target, volatile=True)

            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure error and record loss
            err1, err5 = error(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(err1.item(), input.size(0))
            top5.update(err5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if not silence:
            print('Epoch: {:3d} val   loss {loss.avg:.4f} Err@1 {top1.avg:.4f}'
                  ' Err@5 {top5.avg:.4f} Dur {t:4f}'.format(epoch, loss=losses,
                                                 top1=top1, top5=top5, t=time.time()-total_start))

        return losses.avg, top1.avg, top5.avg
