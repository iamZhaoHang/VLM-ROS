#!/usr/bin/env python3

import rospy
from std_msgs.msg import String, Bool
from pathlib import Path
import sounddevice as sd
import sherpa_onnx

class KeywordSpotterNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("keyword_spotter", anonymous=True)

        # Load parameters from ROS parameter server
        self.tokens = rospy.get_param("~tokens", "")
        self.encoder = rospy.get_param("~encoder", "")
        self.decoder = rospy.get_param("~decoder", "")
        self.joiner = rospy.get_param("~joiner", "")
        self.keywords_file = rospy.get_param("~keywords_file", "")
        self.num_threads = rospy.get_param("~num_threads", 1)
        self.provider = rospy.get_param("~provider", "cpu")
        self.max_active_paths = rospy.get_param("~max_active_paths", 4)
        self.keywords_score = rospy.get_param("~keywords_score", 1.0)
        self.keywords_threshold = rospy.get_param("~keywords_threshold", 0.25)
        self.num_trailing_blanks = rospy.get_param("~num_trailing_blanks", 1)
        self.sample_rate = rospy.get_param("~sample_rate", 16000)  # Default to 16kHz
        self.samples_per_read = int(0.1 * self.sample_rate)  # 0.1 second = 100ms

        # State to control audio input
        self.audio_paused = False  # Default is not paused

        # Check that required files exist
        self.assert_file_exists(self.tokens, "tokens")
        self.assert_file_exists(self.encoder, "encoder")
        self.assert_file_exists(self.decoder, "decoder")
        self.assert_file_exists(self.joiner, "joiner")
        self.assert_file_exists(self.keywords_file, "keywords_file")

        # Initialize the keyword spotter
        self.keyword_spotter = sherpa_onnx.KeywordSpotter(
            tokens=self.tokens,
            encoder=self.encoder,
            decoder=self.decoder,
            joiner=self.joiner,
            num_threads=self.num_threads,
            max_active_paths=self.max_active_paths,
            keywords_file=self.keywords_file,
            keywords_score=self.keywords_score,
            keywords_threshold=self.keywords_threshold,
            num_trailing_blanks=self.num_trailing_blanks,
            provider=self.provider,
        )
        self.stream = self.keyword_spotter.create_stream()

        # Publisher for keyword detection results
        self.result_pub = rospy.Publisher("keyword_result", String, queue_size=10)

        # Subscriber for controlling audio input
        rospy.Subscriber("audio_playing", Bool, self.audio_playing_callback)

        rospy.loginfo("Keyword Spotter Node initialized successfully.")

    def assert_file_exists(self, filepath, param_name):
        """Check if a file exists."""
        if not Path(filepath).is_file():
            rospy.logerr(f"Required file for parameter '{param_name}' does not exist: {filepath}")
            rospy.signal_shutdown(f"Missing required file: {filepath}")

    def audio_playing_callback(self, msg):
        """Callback function to handle audio_playing topic."""
        self.audio_paused = msg.data  # Update the audio paused state
        state = "paused" if self.audio_paused else "resumed"
        rospy.loginfo(f"Audio input {state} based on audio_playing topic.")

    def audio_callback(self, indata, frames, time, status):
        """Callback function for processing audio input."""
        if status:
            rospy.logwarn(f"Audio input error: {status}")

        # If audio is paused, skip processing
        if self.audio_paused:
            return

        # Process the audio input
        audio_data = indata.reshape(-1)  # Flatten the audio data
        self.stream.accept_waveform(self.sample_rate, audio_data)

        # Perform decoding
        while self.keyword_spotter.is_ready(self.stream):
            self.keyword_spotter.decode_stream(self.stream)

        # Get keyword spotting result
        result = self.keyword_spotter.get_result(self.stream)
        if result:
            # Publish the keyword spotting result
            msg = String()
            msg.data = result
            self.result_pub.publish(msg)
            rospy.loginfo(f"Keyword spotted: {result}")

    def run(self):
        """Start capturing audio and performing keyword spotting."""
        rospy.loginfo("Starting audio input stream...")
        with sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=self.sample_rate,
            callback=self.audio_callback,
        ):
            rospy.spin()


if __name__ == "__main__":
    try:
        node = KeywordSpotterNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Keyword Spotter Node.")
