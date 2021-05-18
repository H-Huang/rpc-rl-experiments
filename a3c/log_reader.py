from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator("./runs/May14_20-39-27_q2-st-p38xlarge-1/events.out.tfevents.1621024767.q2-st-p38xlarge-1",
            size_guidance={ 
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })
ea.Reload()

print(ea.Tags())

loss_metric_name = 'Train_0/Loss'
reward_metric_name = 'Train_0/Reward'

print(len(ea.Scalars(loss_metric_name)))
print(ea.Scalars(reward_metric_name)[0])