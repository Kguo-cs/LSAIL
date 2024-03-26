import traci
import numpy as np


class Visualizer():

    def __init__(self):
        pass

    def vis(self,simulated_centroid,type_array,date_index,real_time):
        begin=real_time/1000.0
        sumoCmd = ["/usr/bin/sumo-gui",
                   "-S", "True",
                   "-a", './gail_sim/data/meta/sumo/type.add.xml',
                   "-n", "./gail_sim/data/meta/networks/"+str(date_index)+".net.xml",
                   "-b", str(begin),  # begin time
                   "-e", str(begin+20),
                   "--step-length", "0.4",
                   "--time-to-teleport", "-1",
                   "--collision.action", "none",
                   "--default.emergencydecel", "-1",
                   "-W", "true"
                   ]

        traci.start(sumoCmd)

        type_array=np.array(['','bus', 'car','HeavyVehicle','MediumVehicle','motorcycle','taxi'])[type_array]

        agent_num=len(type_array)

        for typeID, vehID in zip(type_array, range(agent_num)):
            traci.vehicle.add(vehID=str(vehID), routeID='', typeID=typeID)
            traci.vehicle.setSpeedMode(str(vehID), 0)  # strictly follow speed control commands
            traci.vehicle.setLaneChangeMode(str(vehID), 0)  # disable auto lane change
            traci.vehicle.moveToXY(vehID=str(vehID), edgeID="", lane="0", x=str(0),
                                   y=str(0), keepRoute=2)  # , angle=str(phi_sumo),

        for i in range(0,len(simulated_centroid)):

            for vehID in range(agent_num):

                pos=simulated_centroid[i][vehID]

                traci.vehicle.moveToXY(vehID=str(vehID), edgeID="", lane="0", x=str(pos[0]),
                                       y=str(pos[1]), keepRoute=2)  # , angle=str(phi_sumo),

            traci.simulationStep()

        traci.close()

