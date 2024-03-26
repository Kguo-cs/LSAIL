import torch.nn as nn
import torch
from .lqr_solver import LQR_solver

class LQR(nn.Module):
    def __init__(self, cfg):
        super(LQR, self).__init__()

        future_num_frames = cfg['future_num_frames']

        acc_w = cfg["acc_w"]

        control_w = cfg["control_w"]

        step_time=cfg['step_time']

        self.n_ctrl = 2

        if control_w==0:
            n=2
        else:
            n=3

        self.n_state = n*self.n_ctrl

        C = torch.zeros([future_num_frames + 1, self.n_state+self.n_ctrl, self.n_state+self.n_ctrl])#

        C[1:, 0, 0] = 1
        C[1:, 1, 1] = 1

        C[:, self.n_ctrl*2, self.n_ctrl*2] = acc_w
        C[:, self.n_ctrl*2+1, self.n_ctrl*2+1] = acc_w

        if control_w!=0:
            C[1:, self.n_ctrl*3, self.n_ctrl*3] = control_w
            C[1:, self.n_ctrl*3+1, self.n_ctrl*3+1] = control_w

        F = torch.zeros([self.n_state, self.n_state+self.n_ctrl])

        for i in range(self.n_ctrl*n):
            F[i][i] = 1
            F[i][i + self.n_ctrl] = step_time

        for i in range(self.n_ctrl*(n-1)):
            F[i][i + self.n_ctrl*2] =  step_time*step_time

        if control_w!=0:
            for i in range(self.n_ctrl):
                F[i][i + self.n_ctrl*3] = step_time * step_time

            F[:,-self.n_ctrl:]/=step_time

        self.F=F[None][None]

        self.C=C[:,None]


        self.LQR_solver=LQR_solver(
            n_state=self.n_state,
            n_ctrl=self.n_ctrl,
            lqr_iter=1,
            T=future_num_frames + 1,
            max_linesearch_iter=10
            # u_lower=-100,
            # u_upper=100,
            )

    def forward(self,x_init,last_action_preds):

        target = last_action_preds[..., :self.n_ctrl]

        plan = self.solve(target, x_init)

        return plan

    def solve(self,target,x_init):
        n_batch,t,n_ctrl=target.shape

        T=t+1

        C=self.C.repeat(1,n_batch,1,1).to(x_init.device)

        c = torch.zeros([T, n_batch, self.n_state + self.n_ctrl]).to(x_init.device)  # target_state

        c[1:,:,:n_ctrl]=-target.permute(1,0,2)

        #c[1:,:,n_ctrl-1]*=self.yaw_w

        F=self.F.repeat(T,n_batch,1,1).to(x_init.device)

        x_lqr=self.LQR_solver.solve(x_init,C,c,F)

        return x_lqr[1:,:,:n_ctrl].permute(1,0,2)


