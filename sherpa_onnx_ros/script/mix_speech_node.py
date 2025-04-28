#!/usr/bin/env python3

import rospy
from std_msgs.msg import String, Bool, Float32MultiArray
from pathlib import Path
import sherpa_onnx
import numpy as np
from ai_module.srv import ProcessImage  # Import ProcessImage service

class HybridSpeechNode:
    def __init__(self):
        rospy.init_node("mix_speech_node", anonymous=True)

        # --- 加载参数 (Keyword Spotter 部分) ---
        self.tokens_kws = rospy.get_param("~tokens_kws", "")
        self.encoder_kws = rospy.get_param("~encoder_kws", "") # Keyword Spotter 专用 encoder
        self.decoder_kws = rospy.get_param("~decoder_kws", "") # Keyword Spotter 专用 decoder
        self.joiner_kws = rospy.get_param("~joiner_kws", "") # Keyword Spotter 专用 joiner
        self.keywords_file = rospy.get_param("~keywords_file", "")
        self.num_threads = rospy.get_param("~num_threads", 1)
        self.provider = rospy.get_param("~provider", "cpu")
        self.max_active_paths = rospy.get_param("~max_active_paths", 4)
        self.keywords_score = rospy.get_param("~keywords_score", 1.0)
        self.keywords_threshold = rospy.get_param("~keywords_threshold", 0.25)
        self.num_trailing_blanks = rospy.get_param("~num_trailing_blanks", 1)

        # --- 加载参数 (Speech Recognizer 部分) ---
        self.tokens_asr = rospy.get_param("~tokens_asr", "")
        self.encoder_asr = rospy.get_param("~encoder_asr", "") # ASR 专用 encoder
        self.decoder_asr = rospy.get_param("~decoder_asr", "") # ASR 专用 decoder
        self.joiner_asr = rospy.get_param("~joiner_asr", "") # ASR 专用 joiner
        self.rule2_min_trailing_silence = rospy.get_param("~rule2_min_trailing_silence", "")
        self.decoding_method = rospy.get_param("~decoding_method", "greedy_search")
        self.awake = rospy.get_param("~awake", "true") # 沿用 awake 参数控制 ASR 部分
        self.hw_aec = rospy.get_param("~hw_aec", "false") # 沿用 hw_aec 参数

        self.sample_rate = rospy.get_param("~sample_rate", 16000) # 共用 sample_rate 参数

        # 检查文件是否存在 (为 Keyword Spotter 和 Speech Recognizer 分别检查)
        self.assert_file_exists(self.tokens_kws, "tokens_kws")
        self.assert_file_exists(self.encoder_kws, "encoder_kws")
        self.assert_file_exists(self.decoder_kws, "decoder_kws")
        self.assert_file_exists(self.joiner_kws, "joiner_kws")
        self.assert_file_exists(self.keywords_file, "keywords_file")
        
        self.assert_file_exists(self.tokens_asr, "tokens_asr")
        self.assert_file_exists(self.encoder_asr, "encoder_asr")
        self.assert_file_exists(self.decoder_asr, "decoder_asr")
        self.assert_file_exists(self.joiner_asr, "joiner_asr")

        # --- 初始化关键词检测器 ---
        self.keyword_spotter = sherpa_onnx.KeywordSpotter(
            tokens=self.tokens_kws,
            encoder=self.encoder_kws,
            decoder=self.decoder_kws,
            joiner=self.joiner_kws,
            num_threads=self.num_threads,
            max_active_paths=self.max_active_paths,
            keywords_file=self.keywords_file,
            keywords_score=self.keywords_score,
            keywords_threshold=self.keywords_threshold,
            num_trailing_blanks=self.num_trailing_blanks,
            provider=self.provider,
        )
        self.keyword_stream = self.keyword_spotter.create_stream() # Keyword Spotter 专用 stream

        # --- 初始化 Speech Recognizer ---
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=self.tokens_asr,
            encoder=self.encoder_asr, # 使用 Speech Recognizer 专用模型
            decoder=self.decoder_asr, # 使用 Speech Recognizer 专用模型
            joiner=self.joiner_asr, # 使用 Speech Recognizer 专用模型
            num_threads=1, # Speech Recognizer 部分线程数可以单独控制，这里默认 1
            sample_rate=self.sample_rate,
            feature_dim=80, # 保持默认
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4, # 保持默认
            rule2_min_trailing_silence=self.rule2_min_trailing_silence,
            rule3_min_utterance_length=300, # 保持默认
            decoding_method=self.decoding_method,
            provider=self.provider,
        )
        self.recognizer_stream = self.recognizer.create_stream() # Speech Recognizer 专用 stream

        # --- ROS 发布者、订阅者和客户端 ---
        self.keyword_result_pub = rospy.Publisher("keyword_result", String, queue_size=10) # 发布关键词检测结果
        self.audio_playing_pub = rospy.Publisher("audio_playing", Bool, queue_size=10) # 沿用 audio_playing 发布
        rospy.Subscriber("audio_playing", Bool, self.audio_playing_callback) # 沿用 audio_playing 订阅
        rospy.Subscriber("kw_awake", Bool, self.keywords_awake_callback) # 沿用 kw_awake 订阅
        rospy.Subscriber("audio_raw_data", Float32MultiArray, self.audio_data_callback) # 订阅音频数据
        self.process_image_client = rospy.ServiceProxy('process_image', ProcessImage) # 沿用 process_image 服务客户端

        # --- 状态控制 ---
        self.audio_paused = False # 沿用 audio_paused
        self.awake_asr = self.awake #  为 ASR 部分单独维护 awake 状态，初始值与全局 awake 参数一致

        rospy.loginfo("Hybrid Speech Node initialized successfully.")

    def assert_file_exists(self, filepath, param_name):
        """检查文件是否存在 (复用)."""
        if not Path(filepath).is_file():
            rospy.logerr(f"Required file for parameter '{param_name}' does not exist: {filepath}")
            rospy.signal_shutdown(f"Missing required file: {filepath}")

    def audio_playing_callback(self, msg):
        """处理 audio_playing topic 的回调函数 (复用)."""
        self.audio_paused = msg.data
        state = "paused" if self.audio_paused else "resumed"
        rospy.loginfo(f"Audio input {state} based on audio_playing topic.")

    def keywords_awake_callback(self, msg):
        """处理 kw_awake topic 的回调函数 (复用，但修改为控制 ASR 部分的 awake 状态)."""
        self.recognizer_stream = self.recognizer.create_stream() # 清空 ASR stream 缓冲区
        self.awake_asr = msg.data # 更新 ASR 部分的 awake 状态
        state = "awake" if self.awake_asr else "sleep"
        rospy.loginfo(f"ASR awake state changed to {state} based on kw_awake topic.")


    def audio_data_callback(self, msg):
        """处理 audio_raw_data topic 的回调函数 (合并 Keyword Spotter 和 Speech Recognizer 功能)."""
        if self.audio_paused:
            return

        audio_data = np.array(msg.data, dtype=np.float32)

        # --- 1. Keyword Spotting ---
        self.keyword_stream.accept_waveform(self.sample_rate, audio_data)
        while self.keyword_spotter.is_ready(self.keyword_stream):
            self.keyword_spotter.decode_stream(self.keyword_stream)

        keyword_result = self.keyword_spotter.get_result(self.keyword_stream)
        if keyword_result:
            msg = String()
            msg.data = keyword_result
            self.keyword_result_pub.publish(msg)
            rospy.loginfo(f"Keyword Spotted: {keyword_result}")
            self.keyword_stream = self.keyword_spotter.create_stream() # 检测到关键词后重置 Keyword Stream
            self.recognizer_stream = self.recognizer.create_stream() # 同时为了避免干扰，也重置 ASR Stream
            return # 检测到关键词后，直接返回，不再进行后续的 Speech Recognition

        # --- 2. Speech Recognition (只有在没有检测到关键词的情况下才执行) ---
        if self.awake_asr: # 只有在 ASR 部分 awake 的时候才进行语音识别
            self.recognizer_stream.accept_waveform(self.sample_rate, audio_data)
            while self.recognizer.is_ready(self.recognizer_stream):
                self.recognizer.decode_stream(self.recognizer_stream)

            if self.recognizer.is_endpoint(self.recognizer_stream): # 检查 ASR 是否 endpoint
                asr_result = self.recognizer.get_result(self.recognizer_stream)
                if asr_result:
                    rospy.loginfo(f"Speech Recognition Result: {asr_result}")
                    try:
                        process_image_request = ProcessImage._request_class()
                        process_image_request.prompt = asr_result
                        process_image_response = self.process_image_client.call(process_image_request)
                        rospy.loginfo(f"Called process_image service with prompt: '{asr_result}'")
                    except rospy.ServiceException as e:
                        rospy.logerr(f"Failed to call process_image service: {e}")
                self.recognizer_stream = self.recognizer.create_stream() # 处理完一个 utterance 后重置 ASR Stream


    def run(self):
        """运行混合语音节点."""
        rospy.loginfo("Hybrid Speech Node running, waiting for audio data...")
        rospy.spin()

if __name__ == "__main__":
    try:
        node = HybridSpeechNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Hybrid Speech Node.")