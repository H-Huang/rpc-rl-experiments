from enum import Enum

class ExecutionMode(Enum):
    grpc = "grpc"
    cpu_rpc = "cpu_rpc"
    cuda_rpc = "cuda_rpc"
    cuda_rpc_with_batch = "cuda_rpc_with_batch"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)
