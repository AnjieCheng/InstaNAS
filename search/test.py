import argparse
import torch
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import numpy as np
import tqdm
import time
import utils
import os

import torch.backends.cudnn as cudnn

cudnn.benchmark = True
torch.set_num_threads(1)


from imagenet_seq import data

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='InstaMobile_C10')
parser.add_argument('--data_dir', default='../data/')
parser.add_argument('--load', default=None)
args = parser.parse_args()


def test():

    total_ops, total_lat = [], []
    matches, policies = [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        with torch.no_grad():
            inputs, targets = Variable(inputs).cpu(), Variable(targets).cpu()

        start = time.time()
        #--------------------------------------------------------------------------------------------#
        with torch.no_grad():
            probs, _ = agent(inputs)
            policy = probs.clone()

            policy[policy < 0.5] = 0.0
            policy[policy >= 0.5] = 1.0

            preds  = instanet.module.forward_single(inputs, policy.data.squeeze(0))
        #--------------------------------------------------------------------------------------------#
        end = time.time()

        _, pred_idx = preds.max(1)
        match = (pred_idx == targets).data.float()

        matches.append(match)
        policies.append(policy.data)

        total_ops.append(ops)
        total_lat.append(end-start)

    accuracy, _, sparsity, variance, policy_set = utils.performance_stats(
        policies, matches, matches)
    ops_mean, ops_std = np.mean(total_ops), np.std(total_ops)
    lat_mean, lat_std = np.mean(total_lat), np.std(total_lat)

    log_str = u'''
    Accuracy: %.3f
    Block Usage: %.3f \u00B1 %.3f
    FLOPs/img: %.2E \u00B1 %.2E
    Latency/img: %.3f \u00B1 %.3f
    Unique Policies: %d
    ''' % (accuracy, sparsity, variance, ops_mean, ops_std, lat_mean, lat_std, len(policy_set))

    print(log_str)


os.system("NUM_CORES=36; export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES")
testloader = data.Loader('val', batch_size=1, num_workers=32, cuda=False, shuffle=True)

instanet, agent = utils.get_model(args.model)

# if no model is loaded, use all blocks
agent.logit.weight.data.fill_(0)
agent.logit.bias.data.fill_(10)

print("  + loading checkpoints")
if args.load is not None:
    checkpoint = torch.load(args.load)
    # instanet.load_state_dict(checkpoint['instanet'])
    new_state = instanet.state_dict()
    new_state.update(checkpoint['instanet'])
    instanet.load_state_dict(new_state)
    agent.load_state_dict(checkpoint['agent'], load_from_blockdrop=False)


instanet.eval().cpu()
agent.eval().cpu()

test()
