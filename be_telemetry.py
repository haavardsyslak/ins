import asyncio
import threading
import time
import base64
import logging
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
        self._setup()
        self.foxglove_bridge = FoxgloveBridge(telem_names=self.telemetry_messages)
        self.mcap_logger = McapLogger(filename, telemetry_names=self.telemetry_messages)

    def start_floxglove_bridge(self):
        self.foxglove_bridge.start()

    def start_mcap_logger(self):
        self.mcap_logger.start()

    def start(self):
        self.start_floxglove_bridge()
        self.start_mcap_logger()

    def stop_foxglove_bridge(self):
        self.foxglove_bridge.stop()

    def stop_mcap_logger(self):
        self.mcap_logger.stop()

    def stop(self):
        self.mcap_logger.stop()
        self.stop_foxglove_bridge()

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

    def log_message(self, msg_name, data, timestamp=None):
        mcap_channel_ids = self.foxglove_bridge.channel_ids
        foxglove_channel_ids = self.foxglove_bridge.channel_ids
        if timestamp is None:
            timestamp = time.time_ns()

        if msg_name in mcap_channel_ids:
            chan_id = mcap_channel_ids[msg_name]
            self.mcap_logger.log_message(msg_name, message=data, timestamp=timestamp)

            # self.foxglove_bridge.writer._writer.add_message(
            #     channel_id=chan_id, log_time=timestamp, publish_time=timestamp, data=data)
            # self.foxglove_bridge.writer.write_message(
            #     topic=f"blueye.protocol{msg_name}", message=data, publish_time=timestamp, log_time=timestamp)
        if msg_name in foxglove_channel_ids:
            try:
                asyncio.run(
                    self.foxglove_bridge.server.send_message(
                        foxglove_channel_ids[msg_name], timestamp, data)
                )
            except TypeError as e:
                self.foxglove_bridge.logger.info(
                    f"Error sending message for {msg_name}: {e}")
        else:
            self.foxglove_bridge.logger.info(
                f"Warning: Channel ID not found for message type: {msg_name}")
                


    def parse_be_message(self, payload_msg_name, data):
        channel_ids = self.foxglove_bridge.channel_ids
        if payload_msg_name in channel_ids:

            chan_id = channel_ids[payload_msg_name]
            timestamp = time.time_ns()
            self.mcap_logger.log_message(payload_msg_name, message=data, timestamp=timestamp)
            # self.foxglove_bridge.writer._writer.add_message(
            #     channel_id=chan_id, log_time=timestamp, publish_time=timestamp, data=data)
            # self.foxglove_bridge.writer.write_message(
            #     topic=f"blueye.protocol{payload_msg_name}", message=data, publish_time=timestamp, log_time=timestamp)
            try:
                asyncio.run(
                    self.foxglove_bridge.server.send_message(
                        channel_ids[payload_msg_name], timestamp, data)
                )

            except TypeError as e:
                self.foxglove_bridge.logger.info(
                    f"Error sending message for {payload_msg_name}: {e}")
        else:
            self.foxglove_bridge.logger.info(
                f"Warning: Channel ID not found for message type: {payload_msg_name}")


# Example usage
if __name__ == "__main__":
    telem = DroneTelemetry("asdf.mcap")
    telem.start_mcap_logger()
    telem.start_floxglove_bridge()

    while True:
        inn = input("Stop? [y/N]")
        if inn == "y":
            break
    telem.stop_mcap_logger()
    telem.stop_foxglove_bridge()
