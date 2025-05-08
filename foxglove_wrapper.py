from mcap_protobuf.writer import Writer
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf import descriptor_pb2
from foxglove_schemas_protobuf import LocationFix_pb2
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

# Initialize MCAP file
with open("location.mcap", "wb") as f:
    writer = Writer(f)
    # proto_writer = Writer(writer)

    # Register schema
    schema_name = "foxglove.LocationFix"
    # writer.register_message_type(LocationFix_pb2.LocationFix, schema_name=schema_name)

    # Create LocationFix message
    # timestamp = Timestamp(nanos=int(time.time()*1e9))
    # create a timestamp
    now = time.time()
    secs = int(now)
    nanos = int((now - secs) * 1e9)

    msg = LocationFix_pb2.LocationFix(
        timestamp=Timestamp(seconds=secs, nanos=nanos),
        latitude=63.430515,
        longitude=10.395053,
        altitude=0.0,
    )
    # Optional fields
    # msg.horizontal_accuracy_m = 5.0
    # msg.vertical_accuracy_m = 2.0

    # Write the message to topic
    writer.write_message(
        topic="/LocationFix",                 # this is the topic the Map panel expects
        message=msg,
        log_time=int(time.time() * 1e9),   # nanoseconds
        publish_time=int(time.time() * 1e9),
    )

    writer.finish()


class AsyncFoxgloveServerWrapper:
    def __init__(self, host="0.0.0.0", port=8766, name="Foxglove Python Logger"):
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
        self.server = AsyncFoxgloveServerWrapper(host="0.0.0.0", port=8766,
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

    def register_blueye_descriptors(self):
        descriptors = {}

        # Get the module corresponding to the namespace
        module = sys.modules["blueye.protocol"]

        # Iterate through all the attributes of the module

        for name, obj in inspect.getmembers(module):
            # Check if the object is a class, ends with 'Tel', and has a _meta attribute with pb
            if (
                inspect.isclass(obj)
                and name.endswith("Tel")
                and hasattr(obj, "_meta")
                and hasattr(obj._meta, "pb")
            ):
                try:
                    # Access the DESCRIPTOR
                    descriptor = obj._meta.pb.DESCRIPTOR

                    # Create a FileDescriptorSet
                    file_descriptor_set = descriptor_pb2.FileDescriptorSet()

                    # Add the descriptor and its dependencies
                    add_file_descriptor_and_dependencies(descriptor.file, file_descriptor_set)

                    # Serialize the FileDescriptorSet to binary
                    serialized_data = file_descriptor_set.SerializeToString()

                    # Base64 encode the serialized data
                    # schema_base64 = base64.b64encode(serialized_data).decode("utf-8")

                    # Store the serialized data in the dictionary
                    descriptors[name] = serialized_data
                except AttributeError as e:
                    print(f"Skipping message: {name}: {e}")
                    # Skip non-message types
                    raise e

    # Register channels and schemas after collecting them
        for message_name, descriptor in descriptors.items():
            topic = f"blueye.protocol.{message_name}"
            schema_name = topic

            channel = {
                "topic": topic,
                "encoding": "protobuf",
                "schemaName": schema_name,
                "schema": base64.b64encode(descriptor).decode("utf-8"),
            }
            chan_id = self.server.add_channel(channel)
            # Store the chan_id in the map
            self.server_channels[topic] = chan_id

            # Register the schema with the writer
            schema_id = self.writer._writer.register_schema(
                name=schema_name,
                encoding="protobuf",
                data=descriptor,
            )

            self.schemas[topic] = schema_id

            chan_id = self.writer._writer.register_channel(
                schema_id=schema_id,
                topic=topic,
                message_encoding="protobuf",
            )
            self.topics.append(topic)
            self.mcap_channels[topic] = chan_id

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

    def register_topic(self, message, topic, schema_name):

        if schema_name is None:
            schema_name = topic

        self.topics.append(topic)

        descriptor = self.get_protobuf_descriptor(message)
        # schema_base64 = base64.b64encode(descriptor).decode("utf-8")
        message_name = message.__class__
        # Register the schema with the writer
        schema_id = self.writer._writer.register_schema(
            name=schema_name,
            encoding="protobuf",
            data=descriptor,
        )
        self.schemas[topic] = schema_id

        chan_id = self.writer._writer.register_channel(
            schema_id=schema_id, topic=topic, message_encoding="protobuf"
        )
        self.mcap_channels[topic] = chan_id

        schema_id = self.schemas[topic]

        channel = {
            "topic": topic,
            "encoding": "protobuf",
            "schemaName": schema_name,
            "schema": base64.b64encode(descriptor).decode("utf-8"),
        }

        if self.server:
            self.server_channels[topic] = self.server.add_channel(channel)

        print(self.server_channels)

    def publish(self, topic: str, msg: ProtoMessage, timestamp_ns: int = None, schema_name=None):
        timestamp_ns = timestamp_ns or self._get_timestamp_ns()

        if topic not in self.topics:
            self.register_topic(msg, topic, schema_name)

        # MCAP write
        # if type(msg) == bytes:
        #     self.writer._writer.add_message(
        #         channel_id=self.mcap_channels[topic],
        #         data=msg,
        #         log_time=timestamp_ns,
        #         publish_time=timestamp_ns,
        #     )

        # else:
        self.writer.write_message(
            topic=topic,
            message=msg,
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
        )

        # Live stream
        if self.server:
            channel_id = self.server_channels[topic]
            self.server.send_message(channel_id, timestamp_ns, msg.SerializeToString())

    def close(self):
        if self.server:
            self.server.stop()
        self.writer.finish()
        self.f.close()

