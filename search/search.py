import argparse
import os
import time
from datetime import datetime
import math
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import utils
import torch.optim as optim
from torch.distributions import Bernoulli

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.set_num_threads(1)

import warnings
warnings.filterwarnings("ignore")

"""
Set seed
"""
np.random.seed = 8888
torch.manual_seed(8888)
torch.cuda.manual_seed(8888)
torch.cuda.manual_seed_all(8888)

parser = argparse.ArgumentParser(description='InstaNas Search Stage')
parser.add_argument('--static_ep', type=float, default=30, help='static reward target epoch')
parser.add_argument('--static', type=bool, default=False, help='use static reward')
parser.add_argument('--batch_iter', type=bool, default=False, help='batch-wise iterative training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--net_lr', type=float, default=None, help='learning rate for net, use `args.lr` if not set')
parser.add_argument('--beta', type=float, default=0.8, help='entropy multiplier')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--model', default='InstaMobile_ImgNet', help='<Net>_<Dataset>')
parser.add_argument('--data_dir', default='../data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--ft_batch_size', type=int, default=32, help='ft batch size')
parser.add_argument('--epoch_step', type=int, default=100, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=100, help='total epochs to run')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')
parser.add_argument('--alpha', type=float, default=0.80, help='probability bounding factor')
parser.add_argument('--train_net_iter', type=int, default=1)
parser.add_argument('--train_agent_iter', type=int, default=1)
parser.add_argument('--pos_w', type=float, default=10.)
parser.add_argument('--neg_w', type=float, default=-10.)
parser.add_argument('--finetune_first', action="store_true", default=False)
parser.add_argument('--lat_exp', type=float, default=1.)
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)


def get_reward(preds, targets, policy, elasped, baseline):

    highest_point = - (lb - ub)*(ub - lb)/4

    sparse_reward = -1 * (elasped.cuda().data - ub) * (elasped.cuda().data - lb) / highest_point
    sparse_reward = torch.clamp(sparse_reward, min=0.)

    _, pred_idx = preds.max(1)
    match = (pred_idx == targets).data

    reward = sparse_reward ** args.lat_exp

    reward[match]   *= args.pos_w
    reward[match==0] = args.neg_w

    reward = reward.unsqueeze(1)
    reward = reward.unsqueeze(2)

    return reward, match.float(), elasped


def train(epoch):

    agent.train()
    instanet.eval()

    matches, rewards, policies, dur, advantages = [], [], [], [], []
    matches_, rewards_, policies_, dur_ = [], [], [], []

    for idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = Variable(inputs).cuda(async=True), Variable(targets).cuda(async=True)

        probs, _ = agent(inputs)
        #---------------------------------------------------------------------#

        policy_map = probs.data.clone()
        policy_map[policy_map < 0.5] = 0.0
        policy_map[policy_map >= 0.5] = 1.0
        policy_map = Variable(policy_map)

        probs = probs*args.alpha + (1-probs)*(1-args.alpha)
        distr = Bernoulli(probs)
        policy = distr.sample()

        with torch.no_grad():
            v_inputs = Variable(inputs.data)

        instanet.eval()
        preds_map, lat_map = instanet.forward(v_inputs, policy_map)
        preds_sample, lat = instanet.forward(v_inputs, policy)

        reward_map, match_map, avg_elasped_map = get_reward(preds_map, targets, policy_map.data, lat_map, instanet.module.baseline)
        reward_sample, match, avg_elasped = get_reward(preds_sample, targets, policy.data, lat, instanet.module.baseline)

        advantage = reward_sample - reward_map

        loss = -distr.log_prob(policy)
        loss = loss * Variable(advantage).expand_as(policy)

        loss = loss.sum()

        probs = probs.clamp(1e-15, 1-1e-15)
        entropy_loss = -probs*torch.log(probs)
        entropy_loss = args.beta*entropy_loss.sum()

        loss = (loss - entropy_loss)/inputs.size(0)


        #------------------------Backprop Controller Loss---------------------#    
        #---------------------------------------------------------------------#
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #---------------------------------------------------------------------#

        if args.batch_iter:
        #------------------------Backprop Meta-Graph Loss---------------------#    
        #---------------------------------------------------------------------#
            instanet.train()
            cur_batch_size = policy.shape[0]
            perm_policy = policy[torch.randperm(cur_batch_size)]
            pm, _ = instanet.forward(v_inputs, perm_policy)
            net_loss = F.cross_entropy(pm, targets)
            optimizer_net.zero_grad()
            net_loss.backward()
            optimizer_net.step()
        #---------------------------------------------------------------------#

        matches.append(match.cpu())
        rewards.append(reward_sample.cpu())
        advantages.append(np.mean(advantage.cpu().numpy()))
        policies.append(policy.data.cpu())
        dur.append(np.mean(avg_elasped.data.cpu().numpy()))

        matches_.append(match_map.cpu())
        rewards_.append(reward_map.cpu())
        policies_.append(policy_map.data.cpu())
        dur_.append(np.mean(avg_elasped_map.data.cpu().numpy()))


    #-------------------------------------------------------------------------------------#
    accuracy, reward, sparsity, policy_set = utils.performance_stats(policies, rewards, matches)
    log_str = 'Prob E: %d | A: %.3f | R: %.4f | S: %.3f | #: %d | D: %.4f ' % (
        epoch, accuracy, reward, sparsity, len(policy_set), np.mean(dur))
    print(log_str)

    scores.append(('{}\t{}\t{}\t{:4f}\t{:4f}\t{:4f}\t{:2E}\t{:3f}\t{:d}\t{:4f}')
                    .format(str(datetime.now()), epoch, 'Prob', ub, lb, accuracy, reward,
                            sparsity, len(policy_set), np.mean(dur)))
    with open(os.path.join(args.cv_dir, 'scores.tsv'), 'w') as f:
        print('\n'.join(scores), file=f)

    #-------------------------------------------------------------------------------------#
    accuracy, reward, sparsity, policy_set = utils.performance_stats(policies_, rewards_, matches_)
    log_str = 'Real E: %d | A: %.3f | R: %.4f | S: %.3f | #: %d | D: %.4f | ADV: %.4f ' % (
        epoch, accuracy, reward, sparsity, len(policy_set), np.mean(dur_), np.mean(advantages))
    print(log_str)

    scores.append(('{}\t{}\t{}\t{:4f}\t{:4f}\t{:4f}\t{:2E}\t{:3f}\t{:d}\t{:4f}')
                    .format(str(datetime.now()), epoch, 'Real', ub, lb, accuracy, reward,
                            sparsity, len(policy_set), np.mean(dur_)))
    with open(os.path.join(args.cv_dir, 'scores.tsv'), 'w') as f:
        print('\n'.join(scores), file=f)


def train_net(epoch):

    agent.eval()
    instanet.train()

    matches_, rewards_, policies_, dur_ = [], [], [], []
    for idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader_ft), total=len(trainloader_ft)):
        inputs, targets = Variable(inputs).cuda(async=True), Variable(targets).cuda(async=True)

        probs, _ = agent(inputs)
        #---------------------------------------------------------------------#

        policy_map = probs.data.clone()
        policy_map = Variable(policy_map)

        probs = probs*args.alpha + (1-probs)*(1-args.alpha)
        distr = Bernoulli(probs)
        policy = distr.sample()

        cur_batch_size = policy.shape[0]
        policy = policy[torch.randperm(cur_batch_size)]

        with torch.no_grad():
            v_inputs = Variable(inputs.data)

        preds_map, lat_map = instanet.forward(v_inputs, policy)

        reward_map, match_map, avg_elasped_map = get_reward(preds_map, targets, policy.data, lat_map, instanet.module.baseline)

        loss = F.cross_entropy(preds_map, targets)

        #---------------------------------------------------------------------#

        optimizer_net.zero_grad()
        loss.backward()
        optimizer_net.step()

        matches_.append(match_map.cpu())
        rewards_.append(reward_map.cpu())
        policies_.append(policy.data.cpu())
        dur_.append(np.mean(avg_elasped_map.data.cpu().numpy()))

    accuracy, reward, sparsity, policy_set = utils.performance_stats(policies_, rewards_, matches_)
    log_str = 'FT E: %d | A: %.3f | R: %.4f | S: %.3f | #: %d | D: %.4f |' % (
        epoch, accuracy, reward, sparsity, len(policy_set), np.mean(dur_))
    print(log_str)

    scores.append(('{}\t{}\t{}\t{:4f}\t{:4f}\t{:4f}\t{:2E}\t{:3f}\t{:d}\t{:4f}')
                    .format(str(datetime.now()), epoch, 'FT', ub, lb, accuracy, reward,
                            sparsity, len(policy_set), np.mean(dur_)))
    with open(os.path.join(args.cv_dir, 'scores.tsv'), 'w') as f:
        print('\n'.join(scores), file=f)


def test(epoch, repro_oneshot=False):

    agent.eval()
    instanet.eval()

    matches, rewards, policies, dur = [], [], [], []
    for _, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        with torch.no_grad():
            inputs, targets = Variable(inputs).cuda(async=True), Variable(targets).cuda(async=True)

        probs, _ = agent(inputs)

        policy = probs.data.clone()
        policy[policy < 0.5] = 0.0
        policy[policy >= 0.5] = 1.0

        if repro_oneshot:
            policy = Variable(torch.ones(inputs.shape[0], instanet.module.num_of_blocks, instanet.module.num_of_actions)).float().cuda()
        else:
            policy = Variable(policy)

        preds, lat = instanet.forward(inputs, policy)

        reward, match, avg_elasped = get_reward(
            preds, targets, policy.data, lat, instanet.module.baseline)

        matches.append(match)
        rewards.append(reward)
        policies.append(policy.data)
        dur.append(np.mean(avg_elasped.data.cpu().numpy()))

    accuracy, reward, sparsity, policy_set = utils.performance_stats(policies, rewards, matches)

    f1 = open(args.cv_dir+'/policies.log', 'w')
    f1.write(str(policy_set))
    f1.close()

    log_str = 'TS - A: %.3f | R: %.4f | S: %.3f | #: %d | D: %.4f |' % (accuracy, reward, sparsity, len(policy_set), np.mean(dur))
    print(log_str)

    scores.append(('{}\t{}\t{}\t{:4f}\t{:4f}\t{:4f}\t{:2E}\t{:3f}\t{:d}\t{:4f}')
                    .format(str(datetime.now()), epoch, 'TS', ub, lb, accuracy, reward,
                            sparsity, len(policy_set), np.mean(dur)))
    with open(os.path.join(args.cv_dir, 'scores.tsv'), 'w') as f:
        print('\n'.join(scores), file=f)


    # save the model
    agent_state_dict = agent.state_dict()
    net_state_dict = instanet.state_dict()

    state = {
        'instanet': net_state_dict,
        'agent': agent_state_dict,
        'epoch': epoch,
        'reward': reward,
        'acc': accuracy,
        'latency': np.mean(dur),
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d_A_%.3f_R_%.2E_S_%.2f_#_%d.t7' %
               (epoch, accuracy, reward, sparsity, len(policy_set)))
    return state


#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
trainloader_ft = torchdata.DataLoader(trainset, batch_size=args.ft_batch_size, shuffle=True, num_workers=16)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=16)

instanet, agent = utils.get_model(args.model)

start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    print(' [*] Loaded agent from', args.load)


instanet.cuda()
agent.cuda()

print(" [*] Baseline Latency: {:4f}".format(instanet.module.baseline))

if args.net_lr is None:
    args.net_lr = args.lr
optimizer_net = optim.SGD(instanet.parameters(), lr=args.net_lr, weight_decay=args.wd)
optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.wd)

lr_scheduler = utils.LrScheduler(optimizer, args.lr, args.lr_decay_ratio, args.epoch_step)

scores = ['timestamp\tepoch\tmode\tub\tlb\taccuracy\treward' #6
            '\tnumOfBlocks\tvariety\tnumOfPolicies\tlatency'] #4

max_recur = 1
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    lr_scheduler.adjust_learning_rate(epoch)

    if args.static:
        epoch_ratio = args.static_ep / args.max_epochs
    else:
        epoch_ratio = epoch / args.max_epochs

    ub_sp = instanet.module.baseline_max * 0.7
    ub_ep = instanet.module.baseline * 0.5
    lb_sp = instanet.module.baseline * 0.5 
    lb_ep = 0
    cur_windows_size = ((ub_sp-lb_sp) - (ub_ep-lb_sp)) * epoch_ratio

    if args.static:
       ub = instanet.module.baseline_max 
       lb = instanet.module.baseline_min
    else:
       ub = (ub_sp) - (ub_sp-ub_ep) * epoch_ratio
       lb = instanet.module.baseline_min


    print(' [*] Current Baseline: {:4f}, MIN: {:4f}, UB: {:4f}, LB: {:4f}'.format(instanet.module.baseline, instanet.module.baseline_min, ub, lb))
    print(" [*] EXP: {}".format(args.cv_dir))

    for i in range(args.train_agent_iter):
        if args.finetune_first and epoch==0:
            break
        print(" [*] Train agent ... ")
        train(epoch)
        result = test(epoch)

    if not args.batch_iter:
        for i in range(args.train_net_iter):
            print(" [*] Fine-tuning ... ")
            train_net(epoch)
            result = test(epoch)
