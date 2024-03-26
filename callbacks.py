from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from collections import defaultdict
from typing import Dict
import numpy as np
from ray.tune import CLIReporter

class MultiAgentDrivingCallbacks(DefaultCallbacks):
    def __init__(self):
        DefaultCallbacks.__init__(self)

        self.ade_val = defaultdict(list)

        self.iter_n=20

        self.iter=0

        self.min_ade=np.nan

    def on_episode_start(
        self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        env_index: int, **kwargs
    ):
        episode.user_data["episode_reward_train"] = defaultdict(list)
        episode.user_data["ade_train"] = defaultdict(list)
        episode.user_data["offroad_train"] = defaultdict(list)

        episode.user_data["episode_reward_val"] = defaultdict(list)
        episode.user_data["ade_val"] = defaultdict(list)
        episode.user_data["offroad_val"] = defaultdict(list)

    def on_episode_step(
        self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        active_keys = base_env.envs[env_index].cur_ids#list(base_env.envs[env_index].vehicles.keys())

        type='_'+base_env.envs[env_index].type
        #print(base_env.envs[env_index].type,"step_e")

        for agent_id in active_keys:
            k = agent_id
            info = episode.last_info_for(k)
            if info:
                episode.user_data["episode_reward"+type][k].append(info["episode_reward"])
                episode.user_data["ade"+type][k].append(info["ade"])
                episode.user_data["offroad"+type][k].append(info["offroad"])

    def on_episode_end(
        self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        **kwargs
    ):
        type='_'+base_env.envs[0].type

        #print(type,len(base_env.envs))
        agent_num=0
        cur_num=0
        real_num=0

        for env in base_env.envs:

            agent_num+=len(env._agent_ids)
            cur_num+=len(env.cur_ids)
            real_num+=len(env.real_ids)

        donerate=1-cur_num/agent_num
        realdone=1-real_num/agent_num

        episode.custom_metrics["donerate"+type]=donerate
        episode.custom_metrics["realdone"+type]=realdone

        for info_k, info_dict in episode.user_data.items():
            if len(info_dict):
                mean_value=float(np.mean([vv for v in info_dict.values() for vv in v]))

                episode.custom_metrics["{}".format(info_k)] = mean_value

                if type=="_val" and info_k=="ade_val":
                    self.iter+=1
                    for key,value in info_dict.items():
                        self.ade_val[key].append(np.mean(value))

        if self.iter == self.iter_n:
            self.iter = 0
            ade_list = []
            for v in self.ade_val.values():
                ade_list.append(np.min(v))
            self.ade_val = defaultdict(list)
            self.min_ade=np.mean(ade_list)
        episode.custom_metrics["minade_val"]=self.min_ade

    def on_train_result(self, algorithm, result: dict, **kwargs):#*,#100 steps

        # present the agent-averaged reward.
        if "ade_val_mean" in result["evaluation"]["custom_metrics"]:
            result["ade_val"] = result["evaluation"]["custom_metrics"]["ade_val_mean"]
            result["offroad_val"] = result["evaluation"]["custom_metrics"]["offroad_val_mean"]
            result["donerate_val"] = result["evaluation"]["custom_metrics"]["donerate_val_mean"]
            result["minade_val"] = result["evaluation"]["custom_metrics"]["minade_val_mean"]

        if "ade_train_mean" in result["custom_metrics"]:
            result["episode_reward_mean"] = result["custom_metrics"]["episode_reward_train_mean"]
            result["ade"] = result["custom_metrics"]["ade_train_mean"]
            result["offroad"] = result["custom_metrics"]["offroad_train_mean"]
            result["donerate"] = result["custom_metrics"]["donerate_train_mean"]
            result["realdone"] = result["custom_metrics"]["realdone_train_mean"]



metric_columns = CLIReporter.DEFAULT_COLUMNS.copy()
progress_reporter = CLIReporter(metric_columns=metric_columns)
progress_reporter.add_metric_column("ade")
progress_reporter.add_metric_column("offroad")
progress_reporter.add_metric_column("donerate")
progress_reporter.add_metric_column("realdone")
progress_reporter.add_metric_column("ade_val")
progress_reporter.add_metric_column("minade_val")
progress_reporter.add_metric_column("offroad_val")
progress_reporter.add_metric_column("donerate_val")
