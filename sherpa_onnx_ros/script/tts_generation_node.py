#!/usr/bin/env python3

import logging
import queue
import threading
import time
import sys
import os
import datetime
import rospkg
import numpy as np
import soundfile as sf

try:
    import rospy
    from std_msgs.msg import String
    from std_msgs.msg import Bool
    from std_msgs.msg import Float32MultiArray # 用于发布音频数据
except ImportError:
    print("Please install ROS Python client library.")
    sys.exit(-1)

import sherpa_onnx

class TTSGenerationNode:
    def __init__(self):
        # Load parameters from ROS - TTS 模型相关的参数
        self.vits_model = rospy.get_param("~vits_model", "")
        self.vits_lexicon = rospy.get_param("~vits_lexicon", "")
        self.vits_tokens = rospy.get_param("~vits_tokens", "")
        self.vits_data_dir = rospy.get_param("~vits_data_dir", "")
        self.vits_dict_dir = rospy.get_param("~vits_dict_dir", "")
        self.tts_rule_fsts = rospy.get_param("~tts_rule_fsts", "")
        self.save_sound = rospy.get_param("~save_sound", False)
        self.sid = rospy.get_param("~sid", 0)
        self.debug = rospy.get_param("~debug", False)
        self.provider = rospy.get_param("~provider", "cpu")
        self.num_threads = rospy.get_param("~num_threads", 1)
        self.speed = rospy.get_param("~speed", 1.0)

        # Initialize audio parameters (这里不再需要 buffer 和播放相关的)
        self.sample_rate = None

        # Initialize save directory and counter
        self.save_dir = None
        self.file_counter = 1

        # Get package path for saving audio files
        self.package_path = rospkg.RosPack().get_path("sherpa_onnx_ros")

        # Load TTS model
        self.tts = self.load_model()

        # 发布音频数据的 topic
        self.audio_pub = rospy.Publisher("tts_audio_output", Float32MultiArray, queue_size=10)

        # Subscribe to the text topic
        text_topic = rospy.get_param("~text_topic", "tts_input")
        rospy.Subscriber(text_topic, String, self.text_callback, queue_size=10)
        rospy.loginfo("TTS Generation node started. Waiting for text input...")


    def load_model(self):
        """Load the TTS model."""
        rospy.loginfo("Loading TTS model...")
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=self.vits_model,
                    lexicon=self.vits_lexicon,
                    data_dir=self.vits_data_dir,
                    dict_dir=self.vits_dict_dir,
                    tokens=self.vits_tokens,
                ),
                provider=self.provider,
                debug=self.debug,
                num_threads=self.num_threads,
            ),
            rule_fsts=self.tts_rule_fsts,
            max_num_sentences=1,
        )
        if not tts_config.validate():
            rospy.logerr("Invalid TTS configuration. Please check the parameters.")
            sys.exit(1)

        tts = sherpa_onnx.OfflineTts(tts_config)
        self.sample_rate = tts.sample_rate # 获取采样率
        rospy.loginfo(f"TTS model loaded. Sample rate: {self.sample_rate}")
        return tts

    def text_callback(self, msg):
        """Callback for the ROS topic to receive text input."""
        rospy.loginfo(f"Received text: {msg.data}")
        self.generate_audio(msg.data) # 调用 generate_audio 而不是 generate_and_play

    def generate_audio(self, text):
        """Generate speech from text and publish the audio."""
        rospy.loginfo("Generating audio...")
        start_time = time.time()
        audio = self.tts.generate(
            text,
            sid=self.sid,
            speed=self.speed,
            callback=self.generated_audio_callback, # 使用回调函数发布音频块
        )
        end_time = time.time()

        if len(audio.samples) == 0:
            rospy.logerr("Error in generating audio.")
            return

        # Save the generated audio if save_sound is enabled
        if self.save_sound:
            self.save_audio(audio.samples, audio.sample_rate)

        elapsed_seconds = end_time - start_time
        audio_duration = len(audio.samples) / audio.sample_rate
        real_time_factor = elapsed_seconds / audio_duration

        rospy.loginfo(f"Elapsed time: {elapsed_seconds:.3f} seconds")
        rospy.loginfo(f"Audio duration: {audio_duration:.3f} seconds")
        rospy.loginfo(f"Real-time factor: {real_time_factor:.3f}")


    def save_audio(self, samples, sample_rate):
        """Save the audio to a file in a directory under the package path."""
        # (保存音频的代码和之前一致)
        base_dir = os.path.join(self.package_path, "saved_audio")
        os.makedirs(base_dir, exist_ok=True)
        if self.save_dir is None:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = os.path.join(base_dir, current_time)
            os.makedirs(self.save_dir, exist_ok=True)
            rospy.loginfo(f"Created directory for saving audio: {self.save_dir}")
        filename = os.path.join(self.save_dir, f"{self.file_counter}.wav")
        sf.write(filename, samples, samplerate=sample_rate, subtype="PCM_16")
        rospy.loginfo(f"Audio saved as {filename}")
        self.file_counter += 1

    def generated_audio_callback(self, samples: np.ndarray, progress: float):
        """This function is called whenever audio samples are generated and publishes them."""
        # 将 NumPy 数组转换为 Float32MultiArray 消息并发布
        audio_msg = Float32MultiArray()
        audio_msg.data = samples.tolist() # 将 NumPy 数组转换为 Python list
        self.audio_pub.publish(audio_msg)
        return 1 # 继续生成

if __name__ == "__main__":
    rospy.init_node("tts_generation_node", anonymous=False)
    try:
        node = TTSGenerationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down TTS generation node.")