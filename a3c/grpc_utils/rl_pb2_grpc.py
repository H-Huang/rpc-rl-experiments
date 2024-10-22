# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import rl_pb2 as rl__pb2


class RL_GRPCStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.get_global_model_state_dict = channel.unary_unary(
                '/rl.RL_GRPC/get_global_model_state_dict',
                request_serializer=rl__pb2.Request.SerializeToString,
                response_deserializer=rl__pb2.Response.FromString,
                )
        self.update_global_model_parameters = channel.unary_unary(
                '/rl.RL_GRPC/update_global_model_parameters',
                request_serializer=rl__pb2.Request.SerializeToString,
                response_deserializer=rl__pb2.Response.FromString,
                )


class RL_GRPCServicer(object):
    """Missing associated documentation comment in .proto file."""

    def get_global_model_state_dict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def update_global_model_parameters(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_RL_GRPCServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'get_global_model_state_dict': grpc.unary_unary_rpc_method_handler(
                    servicer.get_global_model_state_dict,
                    request_deserializer=rl__pb2.Request.FromString,
                    response_serializer=rl__pb2.Response.SerializeToString,
            ),
            'update_global_model_parameters': grpc.unary_unary_rpc_method_handler(
                    servicer.update_global_model_parameters,
                    request_deserializer=rl__pb2.Request.FromString,
                    response_serializer=rl__pb2.Response.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'rl.RL_GRPC', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class RL_GRPC(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def get_global_model_state_dict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/rl.RL_GRPC/get_global_model_state_dict',
            rl__pb2.Request.SerializeToString,
            rl__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def update_global_model_parameters(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/rl.RL_GRPC/update_global_model_parameters',
            rl__pb2.Request.SerializeToString,
            rl__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
