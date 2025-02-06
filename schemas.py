imu_schema = {
    "type": "object",
    "properties": {
        "accelerometer": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "gyroscope": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "magnetometer": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "temperature": {"type": "number"},
    },
}

"""
message DvlVelocity {
    NavigationSensorID sensor_id = 1; // Sensor id
    int32 status = 2; // Vendor-specific status of the DVL
    float delta_time = 3; // Time since last velocity measurement (ms)
    float fom = 4; // Figure of merit, a measure of the accuracy of the velocities (m/s)
    Vector3 velocity = 5; // Velocity, x forward, y left, z down (m/s)
    bool is_water_tracking = 6; // Water tracking status
    repeated DvlTransducer transducers = 7; // List of transducers
    }
"""
dvl_schema = {
    "type": "object",
    "properties": {
        "status": {"type": "integer"},
        "delta_time": {"type": "number"},
        "fom": {"type": "number"},
        "velocity": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "is_water_tracking": {"type": "boolean"},
    },
}

"""
message PositionEstimate {
    float northing = 1; // Position from reset point (m)
    float easting = 2; // Position from reset point (m)
    float heading = 3; // Continuous heading estimate (rad)
    float surge_rate = 4; // Velocity in surge (m/s)
    float sway_rate = 5; // Velocity in sway (m/s)
    float yaw_rate = 6; // Rotaion rate in yaw (rad/s)
    float ocean_current = 7; // Estimated ocean current (m/s)
    float odometer = 8; // Travelled distance since reset (m)
    bool is_valid = 9; // If the estimate can be trusted
    LatLongPosition global_position = 10; // Best estimate of the global position in decimal degrees
    repeated NavigationSensorStatus navigation_sensors = 11; // List of available sensors with status
    float speed_over_ground = 12; // Speed over ground (m/s)
    float course_over_ground = 13; // Course over ground (°)
    int32 time_since_reset_sec = 14; // Time since reset (s)
    }
"""
pos_estimate_schema = {
    "type": "object",
    "properties": {
        "northing": {"type": "number"},
        "easting": {"type": "number"},
        "heading": {"type": "number"},
        "surge_rate": {"type": "number"},
        "sway_rate": {"type": "number"},
        "yaw_rate": {"type": "number"},
        "ocean_current": {"type": "number"},
        "odometer": {"type": "number"},
        "is_valid": {"type": "boolean"},
        "global_position": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
            },
        },
        "speed_over_ground": {"type": "number"},
        "course_over_ground": {"type": "number"},
    },
}

depth_schema = {
    "type": "object",
    "properties": {
            "value": {"type": "number"}
    }
}


state_est_schema = {
    "title": "ROV Data Schema",
    "description": "Schema for representing the velocity and pose of an object in 3D space",
    "type": "object",
    "properties": {
        "velocity": {
            "title": "foxglove.Vector3",
            "description": "3D velocity of the object in space",
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "Velocity along the x-axis"
                },
                "y": {
                    "type": "number",
                    "description": "Velocity along the y-axis"
                },
                "z": {
                    "type": "number",
                    "description": "Velocity along the z-axis"
                }
            },
            "required": ["x", "y", "z"]
        },
        "pose": {
            "title": "foxglove.Pose",
            "description": "A position and orientation for an object or reference frame in 3D space",
            "type": "object",
            "properties": {
                "position": {
                    "title": "foxglove.Vector3",
                    "description": "Point denoting position in 3D space",
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "x coordinate length"
                        },
                        "y": {
                            "type": "number",
                            "description": "y coordinate length"
                        },
                        "z": {
                            "type": "number",
                            "description": "z coordinate length"
                        }
                    },
                    "required": ["x", "y", "z"]
                },
                "orientation": {
                    "title": "foxglove.Quaternion",
                    "description": "Quaternion denoting orientation in 3D space",
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "x value"
                        },
                        "y": {
                            "type": "number",
                            "description": "y value"
                        },
                        "z": {
                            "type": "number",
                            "description": "z value"
                        },
                        "w": {
                            "type": "number",
                            "description": "w value"
                        }
                    },
                    "required": ["x", "y", "z", "w"]
                }
            },
            "required": ["position", "orientation"]
        },
        "euler_angles": {
            "title": "foxglove.Vector3",
            "description": "Euler angles (roll, pitch, yaw) derived from quaternion",
            "type": "object",
            "properties": {
                "roll": {
                    "type": "number",
                    "description": "Rotation around the x-axis"
                },
                "pitch": {
                    "type": "number",
                    "description": "Rotation around the y-axis"
                },
                "yaw": {
                    "type": "number",
                    "description": "Rotation around the z-axis"
                }
            },
            "required": ["roll", "pitch", "yaw"]
        },
        # "gyro_bias": {
        #     "type": "object",
        #     "properties": {
        #         "x": {
        #             "type": "number",
        #             "description": "Rotation around the x-axis"
        #         },
        #         "y": {
        #             "type": "number",
        #             "description": "Rotation around the y-axis"
        #         },
        #         "z": {
        #             "type": "number",
        #             "description": "Rotation around the z-axis"
        #         }
        #     },
        # },
        # "acc_bias": {
        #     "type": "object",
        #     "properties": {
        #         "x": {
        #             "type": "number",
        #             "description": "Rotation around the x-axis"
        #         },
        #         "y": {
        #             "type": "number",
        #             "description": "Rotation around the y-axis"
        #         },
        #         "z": {
        #             "type": "number",
        #             "description": "Rotation around the z-axis"
        #         }
        #     },
        # }
    },
    "required": ["velocity", "pose", "euler_angles"]  # Moved outside "properties"
}
