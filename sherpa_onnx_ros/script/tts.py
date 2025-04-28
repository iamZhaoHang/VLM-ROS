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
except ImportError:
    print("Please install ROS Python client library.")
    sys.exit(-1)

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print("  pip install sounddevice")
    sys.exit(-1)

import sherpa_onnx

class OfflineTTSNode:
    def __init__(self):
        # Load parameters from ROS
        self.vits_model = rospy.get_param("~vits_model", "")
        self.vits_lexicon = rospy.get_param("~vits_lexicon", "")
        self.vits_tokens = rospy.get_param("~vits_tokens", "")
        self.vits_data_dir = rospy.get_param("~vits_data_dir", "")
        self.vits_dict_dir = rospy.get_param("~vits_dict_dir", "")
        self.tts_rule_fsts = rospy.get_param("~tts_rule_fsts", "")
        self.save_sound = rospy.get_param("~save_sound", False)  # 是否保存音频文件
        self.sid = rospy.get_param("~sid", 0)
        self.debug = rospy.get_param("~debug", False)
        self.provider = rospy.get_param("~provider", "cpu")
        self.num_threads = rospy.get_param("~num_threads", 1)
        self.speed = rospy.get_param("~speed", 1.0)

        # Initialize audio parameters
        self.buffer = queue.Queue()
        self.started = False
        self.stopped = False
        self.killed = False
        self.sample_rate = None
        self.event = threading.Event()
        self.first_message_time = None

        # Initialize save directory and counter
        self.save_dir = None
        self.file_counter = 1

        # Get package path for saving audio files
        self.package_path = rospkg.RosPack().get_path("sherpa_onnx_ros")

        # Load TTS model once
        self.tts = self.load_model()

        # Publish /audio_playing topic
        self.audio_playing_pub = rospy.Publisher("audio_playing", Bool, queue_size=10)

        # Subscribe to the text topic
        text_topic = rospy.get_param("~text_topic", "tts_input")
        rospy.Subscriber(text_topic, String, self.text_callback, queue_size=10)
        rospy.loginfo("TTS node started. Waiting for text input...")

    def load_model(self):
        """Load the TTS model only once during initialization."""
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
        self.sample_rate = tts.sample_rate
        rospy.loginfo("TTS model loaded.")
        return tts

    def text_callback(self, msg):
        """Callback for the ROS topic to receive text input."""
        rospy.loginfo(f"Received text: {msg.data}")
        self.generate_and_play(msg.data)

    def generate_and_play(self, text):
        """Generate speech from text and play the audio."""
        # Reset state variables for playback
        self.buffer = queue.Queue()
        self.started = False
        self.stopped = False
        self.killed = False
        self.event.clear()

        # Start a new playback thread
        play_back_thread = threading.Thread(target=self.play_audio)
        play_back_thread.start()

        # Generate audio
        rospy.loginfo("Generating audio...")
        start_time = time.time()
        audio = self.tts.generate(
            text,
            sid=self.sid,
            speed=self.speed,
            callback=self.generated_audio_callback,
        )
        end_time = time.time()

        self.stopped = True

        if len(audio.samples) == 0:
            rospy.logerr("Error in generating audio. Exiting.")
            self.killed = True
            play_back_thread.join()
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

        play_back_thread.join()

    def save_audio(self, samples, sample_rate):
        """Save the audio to a file in a directory under the package path."""
        # Create a base directory for saved audio in the package path
        base_dir = os.path.join(self.package_path, "saved_audio")
        os.makedirs(base_dir, exist_ok=True)

        # Create a directory for the current date and time if it doesn't exist
        if self.save_dir is None:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = os.path.join(base_dir, current_time)
            os.makedirs(self.save_dir, exist_ok=True)
            rospy.loginfo(f"Created directory for saving audio: {self.save_dir}")

        # Save the audio file with a sequential name
        filename = os.path.join(self.save_dir, f"{self.file_counter}.wav")
        sf.write(filename, samples, samplerate=sample_rate, subtype="PCM_16")
        rospy.loginfo(f"Audio saved as {filename}")

        # Increment the file counter
        self.file_counter += 1

    def generated_audio_callback(self, samples: np.ndarray, progress: float):
        """This function is called whenever audio samples are generated."""
        if self.first_message_time is None:
            self.first_message_time = time.time()

        self.buffer.put(samples)

        if not self.started:
            rospy.loginfo("Start playing audio...")
        self.started = True

        # 1 means to keep generating, 0 means to stop generating
        if self.killed:
            return 0

        return 1

    def play_audio_callback(self, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
        """Callback function for playing audio."""
        if self.killed or (self.started and self.buffer.empty() and self.stopped):
            self.event.set()

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

    def play_audio(self):
        """Play audio using the generated audio callback."""
        # Publish audio_playing=True before starting playback
        rospy.loginfo("Audio playback started.")
        self.audio_playing_pub.publish(Bool(data=True))

        try:
            with sd.OutputStream(
                channels=1,
                callback=self.play_audio_callback,
                dtype="float32",
                samplerate=self.sample_rate,
                blocksize=1024,
            ):
                self.event.wait()
        finally:
            # Publish audio_playing=False after playback ends
            rospy.loginfo("Audio playback finished.")
            self.audio_playing_pub.publish(Bool(data=False))


if __name__ == "__main__":
    rospy.init_node("offline_tts_node", anonymous=False)
    try:
        node = OfflineTTSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down TTS node.")
