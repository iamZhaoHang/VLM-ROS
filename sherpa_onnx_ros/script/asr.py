#!/usr/bin/env python3

import rospy
from std_msgs.msg import String, Bool,Float32MultiArray

import numpy as np
import sherpa_onnx
from ai_module.srv import ProcessImage  # Import ProcessImage service

class SpeechRecognizerNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("speech_recognizer", anonymous=True)
        self.awake = rospy.get_param("~awake", "true")
        self.hw_aec = rospy.get_param("~hw_aec", "false")
        # Load parameters
        self.tokens = rospy.get_param("~tokens", "")
        self.encoder = rospy.get_param("~encoder", "")
        self.decoder = rospy.get_param("~decoder", "")
        self.joiner = rospy.get_param("~joiner", "")
        self.rule2_min_trailing_silence = rospy.get_param("~rule2_min_trailing_silence", "")
        self.decoding_method = rospy.get_param("~decoding_method", "greedy_search")
        self.provider = rospy.get_param("~provider", "cpu")
        self.sample_rate = rospy.get_param("~sample_rate", 16000)  # Default to 16kHz

        # State for controlling audio processing
        self.audio_paused = False  # Default state: audio processing is not paused
        self.audio_paused_logged = False  # To track if the "paused" message is logged

        # Create recognizer
        self.recognizer = self.create_recognizer()

        # Initialize stream
        self.stream = self.recognizer.create_stream()

        # Subscriber for controlling audio input
        rospy.Subscriber("audio_playing", Bool, self.audio_playing_callback)

        # Subscriber for controlling audio input
        rospy.Subscriber("kw_awake", Bool, self.keywords_awake_callback)
        rospy.Subscriber("audio_raw_data", Float32MultiArray, self.audio_data_callback) # 订阅音频数据 topic

        # Publisher for recognition results
        # self.result_pub = rospy.Publisher("asr_output", String, queue_size=10)

        # ROS Service Client for process_image
        self.process_image_client = rospy.ServiceProxy('process_image', ProcessImage) # 创建服务客户端
        rospy.loginfo("Waiting for process_image service...")
        rospy.wait_for_service('process_image')
        rospy.loginfo("process_image service started, client initialized.")

        # Log info
        rospy.loginfo("Speech recognizer node started. Listening...")

    def create_recognizer(self):
        """Create a Sherpa ONNX recognizer."""
        rospy.loginfo("Creating recognizer...")
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=self.tokens,
            encoder=self.encoder,
            decoder=self.decoder,
            joiner=self.joiner,
            num_threads=1,
            sample_rate=self.sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=self.rule2_min_trailing_silence,
            rule3_min_utterance_length=300,
            decoding_method=self.decoding_method,
            provider=self.provider,
        )
        rospy.loginfo("Recognizer created successfully.")
        return recognizer

    def audio_playing_callback(self, msg):
        """Callback function to handle /audio_playing topic."""
        self.audio_paused = msg.data
        state = "paused" if self.audio_paused else "resumed"
        rospy.loginfo(f"Audio input {state} based on /audio_playing topic.")

        # Reset the logging state when audio_playing changes
        self.audio_paused_logged = False

    def keywords_awake_callback(self, msg):
        """Callback function to handle /kw_awake topic."""
        self.stream = self.recognizer.create_stream() #clear buffer audio data        
        self.awake = msg.data
        state = "awake" if self.awake else "sleep"
        rospy.loginfo(f"awake {state} based on /kw_awake topic.")
    
    def audio_data_callback(self, msg):
        """Callback function for processing audio input."""
 
        if self.audio_paused:
            return

        # 将接收到的 Float32MultiArray 消息转换为 NumPy 数组
        audio_data = np.array(msg.data, dtype=np.float32)
        self.stream.accept_waveform(self.sample_rate, audio_data)

        # Perform recognition if the recognizer is ready
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        # Check for endpoint and publish result if necessary
        if self.recognizer.is_endpoint(self.stream):
            # Get the final recognition result
            result = self.recognizer.get_result(self.stream)

            # Publish result if it's not empty
            if result:
                msg = String()
                msg.data = result
                # self.result_pub.publish(msg)
                rospy.loginfo(f"Published recognition result: {result}")

                process_image_request = ProcessImage._request_class() # Create a request object
                process_image_request.prompt = result # Set the prompt in the request to the recognition result
                process_image_response = self.process_image_client.call(process_image_request) # Call the service
                rospy.loginfo(f"Called process_image service with prompt: '{process_image_response}'")

            # Reset the recognizer for the next segment
            self.recognizer.reset(self.stream)

    # def audio_callback(self, indata, frames, time, status):
    #     """Callback function for processing audio input."""
    #     if status:
    #         rospy.logwarn(f"Audio input error: {status}")
    #     if not self.awake:
    #         return
    #     # Check if audio input is paused
    #     if self.audio_paused and not self.hw_aec:
    #         if not self.audio_paused_logged:
    #             # rospy.loginfo("Audio input is paused. Skipping audio processing.")
    #             self.audio_paused_logged = True
    #         return

    #     # Reset the logged state if audio is resumed
    #     self.audio_paused_logged = False

    #     # Accept audio samples into the stream
    #     audio_data = np.squeeze(indata)  # Convert to 1D array
    #     self.stream.accept_waveform(self.sample_rate, audio_data)

    #     # Perform recognition if the recognizer is ready
    #     while self.recognizer.is_ready(self.stream):
    #         self.recognizer.decode_stream(self.stream)

    #     # Check for endpoint and publish result if necessary
    #     if self.recognizer.is_endpoint(self.stream):
    #         # Get the final recognition result
    #         result = self.recognizer.get_result(self.stream)

    #         # Publish result if it's not empty
    #         if result:
    #             msg = String()
    #             msg.data = result
    #             # self.result_pub.publish(msg)
    #             rospy.loginfo(f"Published recognition result: {result}")

    #             process_image_request = ProcessImage._request_class() # Create a request object
    #             process_image_request.prompt = result # Set the prompt in the request to the recognition result
    #             process_image_response = self.process_image_client.call(process_image_request) # Call the service
    #             rospy.loginfo(f"Called process_image service with prompt: '{process_image_response}'")

    #         # Reset the recognizer for the next segment
    #         self.recognizer.reset(self.stream)

    def run(self):
        """Start capturing audio and performing recognition."""
        rospy.loginfo("Starting audio input stream...")
        # with sd.InputStream(
        #     samplerate=self.sample_rate,
        #     channels=1,
        #     dtype="float32",
        #     callback=self.audio_callback,
        # ):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = SpeechRecognizerNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down speech recognizer node.")
