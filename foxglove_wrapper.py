from mcap_protobuf.writer import Writer
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf import descriptor_pb2
from foxglove_schemas_protobuf import LocationFix_pb2
from mcap_protobuf.reader import read_protobuf_messages
import inspect
import sys
import time
from typing import Dict, Type, List
from google.protobuf.message import Message as ProtoMessage
from foxglove_websocket.server import FoxgloveServer
from foxglove_websocket.types import Channel
import threading
import asyncio
import base64

class McapProtobufReader:
    def __init__(self, filename):
        self.filename = filename
        self.msg_iter = read_protobuf_messages(filename)

    def get_next_message(self):
        # Use self.msg_iter to get the next message
        try:
            message = next(self.msg_iter)
            return message
        except StopIteration:
            return None

    def __iter__(self):
        return self.msg_iter

class AsyncFoxgloveServerWrapper:
    def __init__(self, host="0.0.0.0", port=8765, name="Foxglove Python Logger"):
        self.server = None
        self.thread = threading.Thread(target=self._start_loop,
                                       args=(host, port, name), daemon=True)
        self.loop_ready = threading.Event()
        self.loop = None
        self.thread.start()
        print("waiging for loop")
        self.loop_ready.wait()  # Wait for loop to be ready

    def _start_loop(self, host, port, name):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        async def start_server():
            self.server = FoxgloveServer(host=host, port=port, name="Blueye SDK bridge")
            print(self.server)
            print("starting server")
            self.server.start()
            print("Server started")
            self.loop_ready.set()

        self.loop.create_task(start_server())
        self.loop.run_forever()

    def add_channel(self, channel: Channel) -> int:
        fut = asyncio.run_coroutine_threadsafe(self.server.add_channel(channel), self.loop)
        return fut.result()

    def send_message(self, channel_id: int, timestamp: int, data: bytes):
        fut = asyncio.run_coroutine_threadsafe(
            self.server.send_message(channel_id, timestamp, data),
            self.loop
        )
        return fut.result()

    # def add_schema(self, schema: Schema) -> int:
    #     fut = asyncio.run_coroutine_threadsafe(
    #         self.server.add_schema(schema),
    #         self.loop
    #     )
    #     return fut.result()

    def stop(self):
        self.server.close()
        self.loop.call_soon_threadsafe(self.loop.stop)


def add_file_descriptor_and_dependencies(file_descriptor, file_descriptor_set):
    """Recursively add descriptors and their dependencies to the FileDescriptorSet"""
    # Check if the descriptor is already in the FileDescriptorSet
    if file_descriptor.name not in [fd.name for fd in file_descriptor_set.file]:
        # Add the descriptor to the FileDescriptorSet
        file_descriptor.CopyToProto(file_descriptor_set.file.add())

        # Recursively add dependencies
        for file_descriptor_dep in file_descriptor.dependencies:
            add_file_descriptor_and_dependencies(file_descriptor_dep, file_descriptor_set)


class FoxgloveLogger:
    def __init__(self, mcap_path: str, stream: bool = True):
        self.f = open(mcap_path, "wb")
        self.writer = Writer(self.f)
        print("FoxgloveLogger initialized")
        self.server = AsyncFoxgloveServerWrapper(host="0.0.0.0", port=8765,
                                                 name="Foxglove Python Logger") if stream else None
        print("FoxgloveLogger")
        self.server_channels: Dict[str, int] = {}  # topic -> channel_id
        self.mcap_channels: Dict[str, int] = {}  # topic -> channel_id
        self.schemas = {}
        self.topics: List[str] = []

    def _get_timestamp_ns(self):
        now = time.time()
        return int(now * 1e9)

    def _register_schema(self, proto_cls: Type[ProtoMessage], type_name: str):
        # Get raw schema bytes
        descriptor = proto_cls.DESCRIPTOR
        schema_bytes = descriptor.file.serialized_pb
        # self.writer.register_message_type(proto_cls, schema_name=type_name)

        self.schemas[type_name] = {
            "proto_cls": proto_cls,
            "type_name": type_name,
        }

    def get_protobuf_descriptor(self, message):
        file_descriptor_set = descriptor_pb2.FileDescriptorSet()
        if hasattr(message, "DESCRIPTOR"):
            descriptor = message.DESCRIPTOR
        elif hasattr(message, "_meta") and hasattr(message._meta, "pb"):
            descriptor = message._meta.pb.DESCRIPTOR
        else:
            raise TypeError(f"Cannot extract descriptor from message of type {type(message)}")

        add_file_descriptor_and_dependencies(descriptor.file, file_descriptor_set)
        serialized = file_descriptor_set.SerializeToString()

        return serialized

    def register_topic(self, message, topic, schema_name=None):
        if schema_name is None:
            schema_name = message.DESCRIPTOR.full_name  

        if topic in self.topics:
            return  

        self.topics.append(topic)

        descriptor = self.get_protobuf_descriptor(message)

        # Register schema for MCAP
        schema_id = self.writer._writer.register_schema(
            name=schema_name,
            encoding="protobuf",
            data=descriptor,
        )
        self.schemas[topic] = schema_id

        # Register channel for MCAP
        chan_id = self.writer._writer.register_channel(
            schema_id=schema_id,
            topic=topic,
            message_encoding="protobuf",
        )
        self.mcap_channels[topic] = chan_id

        # Register channel for live Foxglove streaming
        if self.server:
            channel = {
                "topic": topic,
                "encoding": "protobuf",
                "schemaName": schema_name,
                "schema": base64.b64encode(descriptor).decode("utf-8"),
            }
            server_chan_id = self.server.add_channel(channel)
            self.server_channels[topic] = server_chan_id

    def publish(self, topic: str, msg: ProtoMessage, timestamp_ns: int = None, schema_name=None):
        timestamp_ns = timestamp_ns or self._get_timestamp_ns()

        if topic not in self.topics:
            self.register_topic(msg, topic, schema_name)

        # MCAP write
        self.writer.write_message(
            topic=topic,
            message=msg,
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
        )

        if self.server:
            channel_id = self.server_channels.get(topic)
            if channel_id is not None:
                self.server.send_message(channel_id, timestamp_ns, msg.SerializeToString())
            else:
                print(f"[warning] No channel id for topic {topic}")

    def close(self):
        if self.server:
            self.server.stop()
        self.writer.finish()
        self.f.close()

