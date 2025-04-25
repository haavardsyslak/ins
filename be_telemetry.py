import asyncio
import threading
import time
import base64
import logging
from foxglove import FoxgloveLogger
from mcap_protobuf.writer import Writer
from foxglove_websocket.server import FoxgloveServer
from google.protobuf import descriptor_pb2
from messages_pb2 import CustomTel
import inspect
import sys
import blueye
from blueye.sdk import Drone
import blueye.protocol as bp
from queue import Queue, Empty

from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf import descriptor_pb2
from foxglove_schemas_protobuf import LocationFix_pb2

def add_file_descriptor_and_dependencies(file_descriptor, file_descriptor_set):
    """Recursively add descriptors and their dependencies to the FileDescriptorSet"""
    # Check if the descriptor is already in the FileDescriptorSet
    if file_descriptor.name not in [fd.name for fd in file_descriptor_set.file]:
        # Add the descriptor to the FileDescriptorSet
        file_descriptor.CopyToProto(file_descriptor_set.file.add())

        # Recursively add dependencies
        for file_descriptor_dep in file_descriptor.dependencies:
            add_file_descriptor_and_dependencies(file_descriptor_dep, file_descriptor_set)


def get_protobuf_descriptors(namespace, telem=[]):
    descriptors = {}
    # Get the module corresponding to the namespace
    module = sys.modules[namespace]

    # Iterate through all the attributes of the module
    for name, obj in inspect.getmembers(module):
        # Check if the object is a class, ends with 'Tel', and has a _meta attribute with pb

        if (
            inspect.isclass(obj)
            and name.endswith("Tel")
            and (name in telem or not telem)
        ):
            try:
                # Access the DESCRIPTOR
                if hasattr(obj, "_meta") and hasattr(obj._meta, "pb"):
                    descriptor = obj._meta.pb.DESCRIPTOR
                else:
                    descriptor = obj.DESCRIPTOR

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
                # self.logger.info(f"Skipping message: {name}: {e}")
                # Skip non-message types
                raise e

    return descriptors


class McapLogger:
    def __init__(self, filename: str, telemetry_names):
        self.filename = filename
        self.telem_names = telemetry_names
        self.file = open(filename, "wb")
        self.writer = Writer(self.file)
        self._writer = self.writer._writer
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._log_worker)
        self.queue = Queue()
        self.channel_ids = {}
        self.schemas = {}

        self._running = False

        self.register_channels()

    def register_channels(self):
        # Get Protobuf descriptors for Blueye and custom messages
        blueye_descriptors = get_protobuf_descriptors("blueye.protocol", self.telem_names)
        custom_descriptors = get_protobuf_descriptors("messages_pb2")

        # Register Blueye message channels
        for message_name, schema_base64 in blueye_descriptors.items():
            schema_id = self._writer.register_schema(
                name=f"blueye.protocol.{message_name}",
                encoding="protobuf",
                data=schema_base64
            )

            chan_id = self._writer.register_channel(
                schema_id=schema_id, topic=f"blueye.protocol.{message_name}", message_encoding="protobuf")
            self.channel_ids[message_name] = chan_id
            # self.logger.info(f"Registered topic: blueye.protocol.{message_name}")

        for message_name, schema_base64 in blueye_descriptors.items():
            schema_id = self._writer.register_schema(
                name=f"blueye.protocol.{message_name}",
                encoding="protobuf",
                data=schema_base64
            )

            chan_id = self._writer.register_channel(
                schema_id=schema_id, topic=f"blueye.protocol.{message_name}", message_encoding="protobuf")
            self.channel_ids[message_name] = chan_id

    @property
    def running(self) -> bool:
        return self._running

    def log_message(self, topic: str, message: bytes, timestamp=None):
        """Add a telemetry message to the queue."""
        if timestamp is None:
            timestamp = time.time_ns()

        self.queue.put((topic, message, timestamp))

    def _log_worker(self):
        """Worker thread for writing messages to the MCAP file."""
        while self._running or not self.queue.empty():
            try:
                topic, message, timestamp = self.queue.get(timeout=0.1)
                with self.lock:
                    self.writer._writer.add_message(
                        channel_id=self.channel_ids[topic],
                        log_time=timestamp,
                        publish_time=timestamp,
                        data=message,
                    )
                self.queue.task_done()
            except Empty:
                continue

    def start(self):
        self._running = True
        self.thread.start()

    def stop(self):
        print("Stopping")
        self._running = False
        print("joining")
        self.thread.join()
        print("joined")
        with self.lock:
            self.writer.finish()
            self.file.close()


class FoxgloveBridge:
    def __init__(self, host="0.0.0.0", port=8766, name="Blueye SDK bridge",
                 mcap_file="output.mcap", telem_names=[]):
        self.host = host
        self.port = port
        self.name = name
        self.server = None
        self.channel_ids = {}
        self.mcap_file = mcap_file
        self.telem_names = telem_names
        self.file = mcap_file
        self.thread = None
        self.loop = None
        self.stop_event = asyncio.Event()
        self.logger = logging.getLogger("FoxgloveBridge")
        self.logger.setLevel(logging.DEBUG)

    def _start_bridge(self):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s: [%(levelname)s] <%(name)s> %(message)s"))
        self.logger.addHandler(handler)
        self.logger.info("Starting Foxglove bridge")

        self.logger_sdk = logging.getLogger(blueye.sdk.__name__)
        self.logger_sdk.setLevel(logging.DEBUG)
        self.logger_sdk.addHandler(handler)

    async def start_server(self):
        async with FoxgloveServer(self.host, self.port, self.name) as server:
            self.server = server
            await self.register_channels()
            while not self.stop_event.is_set():
                await asyncio.sleep(1)

            # await self.run()

    async def register_channels(self):
        # Get Protobuf descriptors for Blueye and custom messages
        blueye_descriptors = get_protobuf_descriptors("blueye.protocol", self.telem_names)
        custom_descriptors = get_protobuf_descriptors("messages_pb2")

        # Register Blueye message channels
        for message_name, schema in blueye_descriptors.items():
            schema_base64 = base64.b64encode(schema).decode("utf-8")
            chan_id = await self.server.add_channel(
                {
                    "topic": f"blueye.protocol.{message_name}",
                    "encoding": "protobuf",
                    "schemaName": f"blueye.protocol.{message_name}",
                    "schema": schema_base64,
                }
            )
            self.channel_ids[message_name] = chan_id
            self.logger.info(f"Registered topic: blueye.protocol.{message_name}")

        # Register custom message channels
        for message_name, schema in custom_descriptors.items():

            schema_base64 = base64.b64encode(schema).decode("utf-8")
            chan_id = await self.server.add_channel(
                {
                    "topic": f"custom.{message_name}",
                    "encoding": "protobuf",
                    "schemaName": f"custom.{message_name}",
                    "schema": schema_base64,
                }
            )
            self.channel_ids[message_name] = chan_id
            self.logger.info(f"Registered topic: hs.{message_name}")

    async def run(self):
        while True:
            msg = CustomTel(
                timestamp=time.time(),
                temperature=22.5,
                pressure=1013.25,
                status="OK",
            )

            data = msg.SerializeToString()
            timestamp_ns = time.time_ns()
            # topic = f"custom.{msg.__class__.__name__}"
            # chan_id = self.channel_ids.get(topic)
            chan_id = self.channel_ids[msg.__name__]

            await self.server.send_message(chan_id, timestamp_ns, data)
            # await asyncio.to_thread(self.writer.write_message, channel_id=chan_id, log_time=timestamp_ns, data=data)
            # await asyncio.to_thread(self.writier.write_message, message=msg)
            # else:
            #  logger.warning(f"No channel found for topic {topic}")

            await asyncio.sleep(1)

    def start(self):
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def _run_event_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._start_bridge()
        self.loop.run_until_complete(self.start_server())

    def stop(self):
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.stop_event.set)
            tasks = asyncio.all_tasks(self.loop)
            for task in tasks:
                task.cancel()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            self.logger.info("Foxglove bridge stopped")


class DroneTelemetry:
    def __init__(self, filename: str):
        self.filename = filename
        self.drone = None
        self.callbacks = []
        self.telemetry_messages = []
        self.foxglove_logger = FoxgloveLogger(self.filename)
        # self.foxglove_logger.register_blueye_descriptors()
        self._start_bridge()
        self._setup()

    def _start_bridge(self):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s: [%(levelname)s] <%(name)s> %(message)s"))
        self.logger = logging.getLogger("FoxgloveBridge")
        self.logger.addHandler(handler)
        self.logger.info("Starting Foxglove bridge")

        self.logger_sdk = logging.getLogger(blueye.sdk.__name__)
        self.logger_sdk.setLevel(logging.DEBUG)
        self.logger_sdk.addHandler(handler)

    def _setup(self):
        msgs = []
        self.drone = Drone()
        self.drone.telemetry.set_msg_publish_frequency(bp.CalibratedImuTel, 100)
        msgs.append(bp.CalibratedImuTel)
        self.drone.telemetry.set_msg_publish_frequency(bp.Imu1Tel, 100)
        msgs.append(bp.Imu1Tel)
        self.drone.telemetry.set_msg_publish_frequency(bp.DepthTel, 10)
        msgs.append(bp.DepthTel)
        self.drone.telemetry.set_msg_publish_frequency(bp.DvlVelocityTel, 25)
        msgs.append(bp.DvlVelocityTel)
        self.drone.telemetry.set_msg_publish_frequency(bp.PositionEstimateTel, 10)
        msgs.append(bp.PositionEstimateTel)

        self.drone.telemetry.add_msg_callback(msgs, self.parse_be_message, raw=True)
        self.telemetry_messages = [i.__name__ for i in msgs]

    def to_protobuf_message(self, sdk_msg):
        """
        Convert a Blueye SDK proto-plus message to a native protobuf message.
        """
        if hasattr(sdk_msg, "DESCRIPTOR"):
            return sdk_msg  # already native protobuf

        if hasattr(sdk_msg, "_meta") and hasattr(sdk_msg._meta, "pb"):
            pb_cls = sdk_msg._meta.pb
            pb_msg = pb_cls()
            for field in sdk_msg._meta.fields.values():
                value = getattr(sdk_msg, field.name)
                setattr(pb_msg, field.name, value)
            return pb_msg

        raise TypeError(f"Cannot convert {type(sdk_msg)} to Protobuf")

    def parse_be_message(self, payload_msg_name, data):
        # Now publish the real parsed message
        # self.foxglove_logger.publish(topic, msg, timestamp, topic)

        proto_cls = getattr(blueye.protocol, payload_msg_name)._meta.pb

        # Parse the raw bytes into a Protobuf message
        msg = proto_cls()
        msg.ParseFromString(data)

        # Create a new protobuf message and copy fields
        timestamp = time.time_ns()
        topic = f"blueye.protocol.{msg.__name__}"
        print(topic)
        self.foxglove_logger.publish(topic, msg, timestamp)

        msg = LocationFix_pb2.LocationFix(
            timestamp=Timestamp(seconds=0, nanos=0),
            latitude=63.430515,
            longitude=10.395053,
            altitude=0.0,
        )
        self.foxglove_logger.publish("locationFix", msg, time.time_ns(), "foxglove.LocationFix")



# Example usage
if __name__ == "__main__":
    telem = DroneTelemetry("asdf.mcap")

    while True:
        inn = input("Stop? [y/N]")
        if inn == "y":
            break
    telem.foxglove_logger.close()
