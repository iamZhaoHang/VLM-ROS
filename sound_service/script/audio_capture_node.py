#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray, Bool
import sounddevice as sd
import numpy as np

class AudioCaptureNode:
    def __init__(self):
        rospy.init_node("audio_capture_node", anonymous=True)
        self.sample_rate = rospy.get_param("~sample_rate", 16000)
        self.channels = rospy.get_param("~channels", 1)
        self.dtype = rospy.get_param("~dtype", "float32")
        self.audio_recording_paused = False  # Initially, recording is active
        self.audio_pub = rospy.Publisher("audio_raw_data", Float32MultiArray, queue_size=10)
        rospy.Subscriber("audio_playing", Bool, self.audio_playing_callback)
        rospy.loginfo("Audio Capture Node initialized.")

    def audio_playing_callback(self, msg):
        """Callback function to handle /audio_playing topic."""
        self.audio_recording_paused = msg.data  # Set paused state based on msg.data
        state = "paused" if self.audio_recording_paused else "resumed"
        rospy.loginfo(f"Audio recording {state} based on /audio_playing topic.")

    def audio_callback(self, indata, frames, time, status):
        if status:
            rospy.logwarn(f"Audio input error: {status}")

        if self.audio_recording_paused:
            # If audio recording is paused, skip processing and publishing
            return

        audio_data_msg = Float32MultiArray()
        audio_data_msg.data = indata.flatten().tolist() # 将 NumPy 数组展平并转换为列表
        self.audio_pub.publish(audio_data_msg)

    def run(self):
        rospy.loginfo("Starting audio input stream...")
        with sd.InputStream(channels=self.channels, dtype=self.dtype, samplerate=self.sample_rate, callback=self.audio_callback):
            rospy.spin()

if __name__ == '__main__':
    try:
        node = AudioCaptureNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Audio Capture Node shutting down.")