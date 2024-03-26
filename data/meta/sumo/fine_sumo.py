import numpy as np
import torch


length=np.array([5,5,5.83,12.5,12.5,2.5])


try:

    idm_train_data=np.load("../macro_results/idm_train_data.npy")

except:
    from .get_idm_data import get_data

    idm_train_data = get_data(range(1, 15))#cur_type_id,speed,rel_speed,accel,rel_distance

speed=idm_train_data[:,1]

accel=idm_train_data[:,3]

distance_headway=idm_train_data[:,4]

idm_train_data=idm_train_data[(speed>0) & (np.abs(accel)<15) &(distance_headway>0.1)]

idm_train_data=torch.FloatTensor(idm_train_data).cuda()

try:

    idm_val_data = np.load("../macro_results/idm_val_data.npy")

except:
    from .get_idm_data import get_data

    idm_val_data = get_data(range(1, 15))

speed=idm_val_data[:,1]

accel=idm_val_data[:,3]

distance_headway=idm_val_data[:,4]

idm_val_data=idm_val_data[(speed>0) & (np.abs(accel)<15)& (distance_headway>0.1)]

idm_val_data=torch.FloatTensor(idm_val_data).cuda()

def objective(x,data):

    cur_type_id = data[:, 0].to(int)
    speed = data[:, 1]
    rel_speed = data[:, 2]
    accel = data[:, 3]
    distance_headway = data[:, 4]

    x=x.reshape(-1,5)

    agent_parameter = x[cur_type_id]

    agent_max_speed=agent_parameter[:,0]
    agent_max_acc=agent_parameter[:,1]
    agent_max_dec=agent_parameter[:,2]
    agent_min_gap=agent_parameter[:,3]
    agent_t_tau=agent_parameter[:,4]

    distance_des=agent_min_gap+torch.clamp_min(speed*agent_t_tau+speed*rel_speed/(2*torch.sqrt(agent_max_acc*agent_max_dec)),0)

    idm_acc=agent_max_acc*(1-(speed/agent_max_speed)**4-(distance_des/distance_headway)**2)

    loss=torch.square(idm_acc-accel).mean()

    return loss

max_speed=22.0#, 2.0, 2.0,  2.5, 2
max_acc=2.5
max_dec=4.5
min_gap=2.5
t_tau=1

parameter=[]

for i in range(6):
    parameter.extend([max_speed,max_acc,max_dec,min_gap,t_tau])

parameter=torch.FloatTensor(np.array(parameter)).cuda()

parameter=torch.nn.Parameter(parameter)

optimizer= torch.optim.Adam([parameter], lr=1e-1)

min_loss=1e10

for i in range(1000000):
   # parameter=torch.clamp_min(parameter,min=0.1)

    loss=objective(parameter,idm_train_data)

    with torch.no_grad():
        val_loss = objective(parameter, idm_val_data)
        min_loss = min(min_loss, val_loss)

        print("adam 1e1 mse",loss.detach().data,val_loss,min_loss)
        print(parameter.detach().cpu().numpy().reshape(-1, 5))


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    with torch.no_grad():
        for i in range(6):
            parameter[i*5].clamp_(15, 30)#max speed
            parameter[i * 5+1].clamp_(2.5, 10)# max_acc
            parameter[i * 5+2].clamp_(1, 10)# max_dec
            parameter[i * 5+3].clamp_(0.1, 10)#min_gap
            parameter[i * 5+4].clamp_(0.1, 10)#t_tau

#get result
#