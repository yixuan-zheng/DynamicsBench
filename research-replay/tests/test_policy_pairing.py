"""Paired Adam continuation must preserve momentum without sharing mutable state."""
import copy
import unittest
import torch

class PolicyPairingTests(unittest.TestCase):
    def test_cloned_adam_continuation_and_isolation(self):
        original=torch.tensor([.2,-.3],dtype=torch.float64,requires_grad=True)
        optimizer=torch.optim.Adam([original],lr=.003)
        for _ in range(5):
            optimizer.zero_grad();original.square().sum().backward();optimizer.step()
        saved=copy.deepcopy(optimizer.state_dict())
        clones=[]
        for _ in range(2):
            p=original.detach().clone().requires_grad_(True)
            opt=torch.optim.Adam([p],lr=.003);opt.load_state_dict(copy.deepcopy(saved))
            clones.append((p,opt))
        for p,opt in [(original,optimizer),*clones]:
            opt.zero_grad();p.square().sum().backward();opt.step()
        for p,_ in clones:torch.testing.assert_close(p,original,rtol=0,atol=0)
        before=clones[1][1].state[clones[1][0]]['exp_avg'].clone()
        p,opt=clones[0];opt.zero_grad();(p*3).sum().backward();opt.step()
        torch.testing.assert_close(clones[1][1].state[clones[1][0]]['exp_avg'],before,rtol=0,atol=0)
