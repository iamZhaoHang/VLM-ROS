#!/usr/bin/env python3

import rospy
from std_msgs.msg import String, Bool, Float32MultiArray
from pathlib import Path
import sherpa_onnx
import numpy as np

class KeywordSpotterNode:
    def __init__(self):
        rospy.init_node("keyword_spotter", anonymous=True)

        # 加载参数 (与之前代码相同，但移除 sample_rate 参数)
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
        self.sample_rate = rospy.get_param("~sample_rate", 16000) # 仍然需要 sample_rate 参数给 sherpa_onnx，假设与音频捕获节点一致

        # 检查文件是否存在
        self.assert_file_exists(self.tokens, "tokens")
        self.assert_file_exists(self.encoder, "encoder")
        self.assert_file_exists(self.decoder, "decoder")
        self.assert_file_exists(self.joiner, "joiner")
        self.assert_file_exists(self.keywords_file, "keywords_file")

        # 初始化关键词检测器
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

        # 创建发布者和订阅者
        self.result_pub = rospy.Publisher("keyword_result", String, queue_size=10)
        rospy.Subscriber("audio_playing", Bool, self.audio_playing_callback)
        rospy.Subscriber("audio_raw_data", Float32MultiArray, self.audio_data_callback) # 订阅音频数据 topic

        # 状态控制
        self.audio_paused = False

        rospy.loginfo("Keyword Spotter Node initialized successfully.")

    def assert_file_exists(self, filepath, param_name):
        """检查文件是否存在 (与之前代码相同)."""
        if not Path(filepath).is_file():
            rospy.logerr(f"Required file for parameter '{param_name}' does not exist: {filepath}")
            rospy.signal_shutdown(f"Missing required file: {filepath}")

    def audio_playing_callback(self, msg):
        """处理 audio_playing topic 的回调函数 (与之前代码相同)."""
        self.audio_paused = msg.data
        state = "paused" if self.audio_paused else "resumed"
        rospy.loginfo(f"Audio input {state} based on audio_playing topic.")

    def audio_data_callback(self, msg):
        """处理 audio_raw_data topic 的回调函数."""
            
        if self.audio_paused:
            return

        # 将接收到的 Float32MultiArray 消息转换为 NumPy 数组
        audio_data = np.array(msg.data, dtype=np.float32)

        # 使用 sherpa-onnx 处理音频数据 (与之前代码的 audio_callback 函数相同)
        self.stream.accept_waveform(self.sample_rate, audio_data)

        while self.keyword_spotter.is_ready(self.stream):
            self.keyword_spotter.decode_stream(self.stream)

        result = self.keyword_spotter.get_result(self.stream)
        if result:
            msg = String()
            msg.data = result
            self.result_pub.publish(msg)
            rospy.loginfo(f"Keyword spotted: {result}")

    def run(self):
        """运行关键词检测节点."""
        rospy.loginfo("Keyword Spotter Node running, waiting for audio data...")
        rospy.spin() # 只需要 spin，因为音频输入来自 topic

if __name__ == "__main__":
    try:
        node = KeywordSpotterNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Keyword Spotter Node.")