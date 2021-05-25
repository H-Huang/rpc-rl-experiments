from tensorboard.backend.event_processing import event_accumulator
import statistics
import numpy as np
import collections
import json

def get_stats(values):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    return mean, stdev

def moving_avg(values, n=20):
    # moving values over n time periodos
    cumsum = np.cumsum(np.insert(values, 0, 0)) 
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def process_logs(files):
    loss_metric_name = 'Train_1/Loss'
    load_model_delay = "Train_1/LoadGlobalModelDelay"

    fetch_model_metric = "Train_1/FetchModelDelay"
    update_model_metric = "Train_1/UpdateModel"
    per_episode_time_metric = "Train_1/PerEpisodeTime"

    data = {}
    for experiment, file in files.items():
        ea = event_accumulator.EventAccumulator(f"./runs/{file}",
                size_guidance={ 
                event_accumulator.IMAGES: 4,
                event_accumulator.AUDIO: 4,
                event_accumulator.SCALARS: 0,
                event_accumulator.HISTOGRAMS: 1,
            })
        ea.Reload()
        # print(ea.Tags())
        # print(len(ea.Scalars(load_model_delay)))
        # print(ea.Scalars(load_model_delay)[0])

        # skip beginning data points due to warm up
        experiment_data = {}
        for metric in (fetch_model_metric, update_model_metric, per_episode_time_metric):
            skip_count = 10
            # skip_count = 20
            delay_scalars = ea.Scalars(metric)[skip_count:]

            values = []
            for scalar in delay_scalars:
                values.append(scalar.value)
            mean, stdev = get_stats(values)

            metric_name = metric.split("/")[1]
            
            experiment_data[metric_name] = (mean, stdev)
            # print(f"{experiment=}, {metric_name=}, {mean=}, {stdev=}")

        data[experiment] = experiment_data
        # wall_time, step = time_to_converge(reward_scalars)
        # print(wall_time, step)
    print(json.dumps(data, indent=4))

def calculate_rewards(files):
    reward_metric_name = 'Train_1/Reward'
    for experiment, file in files.items():
        ea = event_accumulator.EventAccumulator(f"./runs/{file}",
                size_guidance={ 
                event_accumulator.IMAGES: 4,
                event_accumulator.AUDIO: 4,
                event_accumulator.SCALARS: 0,
                event_accumulator.HISTOGRAMS: 1,
            })
        ea.Reload()
        # only take last 300 episodes
        reward_scalars = ea.Scalars(reward_metric_name)[-300:]
        reward_values = []
        for scalar in reward_scalars:
            reward_values.append(scalar.value)
        mean, stdev = get_stats(reward_values)
        print(f"{experiment=}, {reward_metric_name=}, {mean=}, {stdev=}")

def time_to_converge(scalars):
    # determine convergence threshold to be 
    threshold = 675
    deque = collections.deque(maxlen=100)
    for scalar in scalars:
        value = scalar.value
        deque.append(value)
        avg = statistics.mean(deque)
        if avg >= threshold:
            return scalar.wall_time, scalar.step
    print("did not converge")
    return None, None

if __name__ == "__main__":
    # reward_files = {
    #     "cuda_rpc_8": "2021-05-18_14:21:41_cuda_rpc_8/events.out.tfevents.1621347702.q3-dy-p3dn24xlarge-2.73506.0",
    #     "cpu_rpc_8": "2021-05-20_05:22:31_cpu_rpc_8/events.out.tfevents.1621574552.q3-dy-p3dn24xlarge-15.88635.0",
    #     "grpc_8": "2021-05-20_16:06:39_grpc_8/events.out.tfevents.1621526799.q3-dy-p3dn24xlarge-1.62428.0",
    # }
    files = {
        # trainrun python main.py --world_size=NUM --execution_mode=grpc --num_episodes=100 
        "grpc_2": "2021-05-20_21:31:28_grpc_2/events.out.tfevents.1621546288.q3-dy-p3dn24xlarge-2.33315.0",
        "grpc_4": "2021-05-20_21:38:23_grpc_4/events.out.tfevents.1621546704.q3-dy-p3dn24xlarge-2.36666.0",
        "grpc_6": "2021-05-20_21:44:03_grpc_6/events.out.tfevents.1621547044.q3-dy-p3dn24xlarge-2.45757.0",
        "grpc_8": "2021-05-20_21:52:48_grpc_8/events.out.tfevents.1621547568.q3-dy-p3dn24xlarge-2.60706.0",
        "grpc_10": "2021-05-21_22:12:10_grpc_10/events.out.tfevents.1621635130.q3-dy-p3dn24xlarge-7.51828.0",
        "grpc_12": "2021-05-21_22:13:07_grpc_12/events.out.tfevents.1621635187.q3-dy-p3dn24xlarge-9.28738.0",
        "grpc_14": "2021-05-21_22:16:36_grpc_14/events.out.tfevents.1621635396.q3-dy-p3dn24xlarge-2.76488.0",
        "grpc_16": "2021-05-21_22:22:11_grpc_16/events.out.tfevents.1621635732.q3-dy-p3dn24xlarge-1.8941.0",
        # trainrun python main.py --world_size=NUM --execution_mode=cpu_rpc --num_episodes=100
        "cpu_rpc_2": "2021-05-20_22:03:35_cpu_rpc_2/events.out.tfevents.1621548215.q3-dy-p3dn24xlarge-2.81600.0",
        "cpu_rpc_4": "2021-05-20_22:08:55_cpu_rpc_4/events.out.tfevents.1621548535.q3-dy-p3dn24xlarge-1.88359.0",
        "cpu_rpc_6": "2021-05-20_22:14:37_cpu_rpc_6/events.out.tfevents.1621548877.q3-dy-p3dn24xlarge-1.735.0",
        "cpu_rpc_8": "2021-05-20_22:20:50_cpu_rpc_8/events.out.tfevents.1621549250.q3-dy-p3dn24xlarge-1.18312.0",
        "cpu_rpc_10": "2021-05-21_21:54:02_cpu_rpc_10/events.out.tfevents.1621634042.q3-dy-p3dn24xlarge-10.11107.0",
        "cpu_rpc_12": "2021-05-21_21:54:50_cpu_rpc_12/events.out.tfevents.1621634090.q3-dy-p3dn24xlarge-2.64739.0",
        "cpu_rpc_14": "2021-05-21_22:02:07_cpu_rpc_14/events.out.tfevents.1621634527.q3-dy-p3dn24xlarge-10.73330.0",
        "cpu_rpc_16": "2021-05-21_22:05:29_cpu_rpc_16/events.out.tfevents.1621634730.q3-dy-p3dn24xlarge-7.45322.0",
        # trainrun python main.py --world_size=NUM --execution_mode=cuda_rpc --num_episodes=100
        "cuda_rpc_2": "2021-05-20_22:26:17_cuda_rpc_2/events.out.tfevents.1621549578.q3-dy-p3dn24xlarge-1.40745.0",
        "cuda_rpc_4": "2021-05-20_22:31:52_cuda_rpc_4/events.out.tfevents.1621549913.q3-dy-p3dn24xlarge-1.42436.0",
        "cuda_rpc_6": "2021-05-20_22:37:41_cuda_rpc_6/events.out.tfevents.1621550261.q3-dy-p3dn24xlarge-1.45580.0",
        "cuda_rpc_8": "2021-05-20_22:44:10_cuda_rpc_8/events.out.tfevents.1621550650.q3-dy-p3dn24xlarge-1.50833.0",
        "cuda_rpc_10": "2021-05-21_21:33:58_cuda_rpc_10/events.out.tfevents.1621632838.q3-dy-p3dn24xlarge-1.64633.0",
        "cuda_rpc_12": "2021-05-21_21:58:14_cuda_rpc_12/events.out.tfevents.1621634294.q3-dy-p3dn24xlarge-8.36174.0",
        "cuda_rpc_14": "2021-05-21_21:48:14_cuda_rpc_14/events.out.tfevents.1621633695.q3-dy-p3dn24xlarge-2.58513.0",
        "cuda_rpc_16": "2021-05-21_18:59:41_cuda_rpc_16/events.out.tfevents.1621623582.q3-dy-p3dn24xlarge-4.52918.0",
        # latest
        "cpu_rpc_16_2": "2021-05-24_20:13:09_cpu_rpc_16/events.out.tfevents.1621887189.q3-dy-p3dn24xlarge-1.10054.0",
        "cpu_rpc_16_3": "2021-05-24_20:29:45_cpu_rpc_16/events.out.tfevents.1621888186.q3-dy-p3dn24xlarge-1.87514.0",
    }
    # calculate_rewards(files)
    process_logs(files)