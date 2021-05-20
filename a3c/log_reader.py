from tensorboard.backend.event_processing import event_accumulator
import statistics
import numpy as np
import collections

# ea = event_accumulator.EventAccumulator("./runs/May14_20-39-27_q2-st-p38xlarge-1/events.out.tfevents.1621024767.q2-st-p38xlarge-1",
#             size_guidance={ 
#             event_accumulator.IMAGES: 4,
#             event_accumulator.AUDIO: 4,
#             event_accumulator.SCALARS: 0,
#             event_accumulator.HISTOGRAMS: 1,
#         })
# ea.Reload()

# print(ea.Tags())

# loss_metric_name = 'Train_0/Loss'
# reward_metric_name = 'Train_0/Reward'

# print(len(ea.Scalars(loss_metric_name)))
# print(ea.Scalars(reward_metric_name)[0])
# # file_name = "May14_20-39-27_q2-st-p38xlarge-1/events.out.tfevents.1621024767.q2-st-p38xlarge-1"

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
    reward_metric_name = 'Train_1/Reward'
    load_model_delay = "Train_1/LoadGlobalModelDelay"

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
        skip_count = 100
        delay_scalars = ea.Scalars(load_model_delay)[skip_count:]

        values = []
        for scalar in delay_scalars:
            values.append(scalar.value)
        mean, stdev = get_stats(values)
        
        print(experiment, mean, stdev)

        reward_scalars = ea.Scalars(reward_metric_name)[skip_count:]
        wall_time, step = time_to_converge(reward_scalars)
        print(wall_time, step)

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
    files = {
        "cuda_rpc_8": "2021-05-18_14:21:41_cuda_rpc_8/events.out.tfevents.1621347702.q3-dy-p3dn24xlarge-2.73506.0",
        "cpu_rpc_8": "2021-05-20_04:44:49_cpu_rpc_8/events.out.tfevents.1621485890.q3-dy-p3dn24xlarge-1.11863.0",
        "grpc_8": "2021-05-19_19:32:04_grpc_8/events.out.tfevents.1621452724.q3-dy-p3dn24xlarge-2.10359.0",
        "grpc_2": "2021-05-16_19:33:13_grpc_2/events.out.tfevents.1621193594.q2-st-p38xlarge-1.10435.0",
    }
    process_logs(files)