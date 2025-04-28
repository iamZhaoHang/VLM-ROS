#!/usr/bin/env python3

import logging
import queue
import time
import sys
import os
import numpy as np

try:
    import rospy
    from std_msgs.msg import String
    from std_msgs.msg import Bool
    from std_msgs.msg import Float32MultiArray # 用于接收音频数据
except ImportError:
    print("Please install ROS Python client library.")
    sys.exit(-1)

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print("  pip install sounddevice")
    sys.exit(-1)


class AudioPlaybackNode:
    def __init__(self):
        # Load parameters from ROS - 音频播放相关的参数
        self.debug = rospy.get_param("~debug", False)
        self.fixed_sample_rate = rospy.get_param("~fixed_sample_rate", 24000)

        # Initialize audio parameters
        self.buffer = queue.Queue()
        self.started_playback = False # Flag to indicate if playback has started
        self.is_playing = False # Flag to track if audio is currently playing
        self.stream = None # OutputStream object

        # Publish /audio_playing topic
        self.audio_playing_pub = rospy.Publisher("audio_playing", Bool, queue_size=1)

        # 订阅音频数据 topic
        audio_topic = rospy.get_param("~audio_topic", "tts_audio_output") # 订阅 tts_audio_output topic
        rospy.Subscriber(audio_topic, Float32MultiArray, self.audio_callback, queue_size=11)

        rospy.loginfo("Audio Playback node started. Using fixed sample rate.") # 修改启动日志


    def audio_callback(self, msg):
        """Callback for receiving audio data and initiating playback."""
        audio_data = np.array(msg.data, dtype=np.float32) # 将接收到的 list 转换为 NumPy 数组
        self.buffer.put(audio_data)

        if not self.started_playback:
            rospy.loginfo("Start playing audio...")
            self.started_playback = True
            if not self.is_playing: # Start playback only if not already playing
                self.start_playback()


    def play_audio_callback(self, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
        """Callback function for playing audio."""
        if self.buffer.empty() and self.started_playback: # Check started_playback to ensure we don't stop prematurely
            outdata.fill(0)
            self.stop_playback() # Stop playback when buffer is empty
            return

        if self.buffer.empty():
            outdata.fill(0)
            return

        n = 0
        while n < frames and not self.buffer.empty():
            remaining = frames - n
            k = self.buffer.queue[0].shape[0]

            if remaining <= k:
                outdata[n:, 0] = self.buffer.queue[0][:remaining]
                self.buffer.queue[0] = self.buffer.queue[0][remaining:]
                n = frames
                if self.buffer.queue[0].shape[0] == 0:
                    self.buffer.get()
                break

            outdata[n : n + k, 0] = self.buffer.get()
            n += k

        if n < frames:
            outdata[n:, 0] = 0


    def start_playback(self):
        """Starts audio playback stream."""
        if self.is_playing: # Prevent starting playback if already playing
            return
        rospy.loginfo("Starting audio output stream.")
        self.audio_playing_pub.publish(Bool(data=True)) # Publish audio_playing=True
        rospy.loginfo("Published audio_playing=True.")
        try:
            self.stream = sd.OutputStream(
                channels=1,
                callback=self.play_audio_callback,
                dtype="float32",
                samplerate=self.fixed_sample_rate, # 使用固定的采样率
                blocksize=1024,
            )
            self.stream.start()
            self.is_playing = True # Set playing flag
            rospy.loginfo("Audio output stream started successfully.")
        except Exception as e:
            rospy.logerr(f"Error starting audio output stream: {e}")
            self.audio_playing_pub.publish(Bool(data=False)) # Ensure audio_playing=False on error
            rospy.loginfo("Published audio_playing=False due to error.")


    def stop_playback(self):
        """Stops audio playback stream."""
        if not self.is_playing: # Prevent stopping if not playing
            return
        rospy.loginfo("Stopping audio output stream.")
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False # Reset playing flag
        self.started_playback = False # Reset started_playback flag for next audio
        self.audio_playing_pub.publish(Bool(data=False)) # Publish audio_playing=False
        rospy.loginfo("Published audio_playing=False.")
        rospy.loginfo("Audio playback finished.")


    def run(self):
        rospy.spin() # Main ROS spin loop


if __name__ == "__main__":
    rospy.init_node("audio_playback_node", anonymous=False)
    try:
        node = AudioPlaybackNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down audio playback node.")