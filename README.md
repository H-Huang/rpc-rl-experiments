# RPC RL Experiments (PyTorch)

Testing an RL application trained using multiple processes and using the [Distributed RPC Framework](https://pytorch.org/docs/master/rpc.html?highlight=rpc). Agent, environment, network taken from the PyTorch [Mario RL Tutorial](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

## Setup

1. Create an environment (conda or virtualenv)
2. Install dependencies using `pip install -r requirements.txt`

## Start

Run `python main.py --world_size=5`. Output will look something like:

```
Rank 2 start
Rank 1 start
Rank 3 start
Rank 4 start
Rank 0 start
Episode 0 - Step 162 - Epsilon 0.9999595008150458 - Mean Reward 989.0 - Mean Length 162.0 - Mean Loss 0.0 - Mean Q Value 0.0 - Time Delta 4.857 - Time 2021-04-23T06:59:33
Episode 10 - Step 2775 - Epsilon 0.9993064905021359 - Mean Reward 733.455 - Mean Length 252.182 - Mean Loss 0.0 - Mean Q Value 0.0 - Time Delta 8.267 - Time 2021-04-23T06:59:57
Episode 20 - Step 4364 - Epsilon 0.998909594787748 - Mean Reward 648.857 - Mean Length 207.81 - Mean Loss 0.0 - Mean Q Value 0.0 - Time Delta 5.002 - Time 2021-04-23T07:00:11
Episode 30 - Step 7643 - Epsilon 0.9980910740820003 - Mean Reward 637.677 - Mean Length 246.548 - Mean Loss 0.0 - Mean Q Value 0.0 - Time Delta 15.799 - Time 2021-04-23T07:00:40
Episode 40 - Step 9653 - Epsilon 0.9975896592455535 - Mean Reward 636.415 - Mean Length 235.439 - Mean Loss 0.0 - Mean Q Value 0.0 - Time Delta 5.003 - Time 2021-04-23T07:00:58
Episode 50 - Step 11464 - Epsilon 0.9971381026996495 - Mean Reward 610.098 - Mean Length 224.765 - Mean Loss 0.281 - Mean Q Value 0.337 - Time Delta 4.68 - Time 2021-04-23T07:01:16
Episode 60 - Step 12321 - Epsilon 0.9969244887186295 - Mean Reward 576.475 - Mean Length 201.934 - Mean Loss 0.34 - Mean Q Value 0.733 - Time Delta 4.653 - Time 2021-04-23T07:01:25
Episode 70 - Step 14708 - Epsilon 0.9963297514279498 - Mean Reward 614.042 - Mean Length 207.113 - Mean Loss 0.37 - Mean Q Value 1.048 - Time Delta 15.797 - Time 2021-04-23T07:01:48
Episode 80 - Step 16846 - Epsilon 0.9957973554046999 - Mean Reward 627.481 - Mean Length 207.938 - Mean Loss 0.386 - Mean Q Value 1.309 - Time Delta 10.252 - Time 2021-04-23T07:02:08
Episode 90 - Step 18304 - Epsilon 0.9954344533661567 - Mean Reward 621.418 - Mean Length 201.132 - Mean Loss 0.395 - Mean Q Value 1.514 - Time Delta 3.705 - Time 2021-04-23T07:02:23
Episode 100 - Step 22003 - Epsilon 0.9945143507382862 - Mean Reward 625.92 - Mean Length 218.38 - Mean Loss 0.412 - Mean Q Value 1.839 - Time Delta 15.015 - Time 2021-04-23T07:02:58
...
```

## Logging

Logs for each run are written to checkpoints/

## Files

`main.py` - Draft implementation of `multi_process()` which uses PyTorch RPC to communicate between workers. There is 1 learner which contains the agent and action-value function, it kicks off multiple actors to interact with the environment. The actors retrieves an action from the leaner and interacts with their individual environments to return an observation containing state, action, reward, next_state (s, a, r, s'). The central learner stores these in a replay buffer and reads from this buffer to train it's model. Architecture is based off of [SEED RL](https://openreview.net/pdf?id=rkgvXlrKwH). File also contains the `single_process()` implementation originally from the tutorial for testing and reference.

`agent.py` - The MarioAgent and MarioNet. Agent uses [Double Deep Q-networks](https://arxiv.org/pdf/1509.06461.pdf) algorithm.

`env_wrappers.py` - Helper function to create and wraps the "SuperMarioBros-1-1-v0" environment using multiple wrappers which preprocess the state output and modify action space.

`metric_logger.py` - MetricLogger used to create graphs and log information during execution.