#!/usr/bin/env python
# coding=utf-8

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer
import numpy as np
import io
from pydub import AudioSegment  # 需要安装 pydub 和 ffmpeg

#联网的TTS模型

# 若没有将API Key配置到环境变量中，需将your-api-key替换为自己的API Key
dashscope.api_key = "sk-#####################################"

model = "cosyvoice-v1"
voice = "longlaotie"

pub = None  # 全局 publisher，在 init_node 后初始化

def tts_callback(data):
    """
    话题 tts_input 的回调函数，接收文本并进行 TTS 处理，发布音频数据。
    """
    global pub
    text = data.data
    rospy.loginfo("Received text: %s", text)

    try:
        synthesizer = SpeechSynthesizer(model=model, voice=voice)
        audio_bytes = synthesizer.call(text)
        rospy.loginfo('[Metric] requestId: {}, first package delay ms: {}'.format(
            synthesizer.get_last_request_id(),
            synthesizer.get_first_package_delay()))

        # 将 MP3 音频数据转换为 Float32MultiArray
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        raw_samples = audio_segment.get_array_of_samples()
        float_samples = np.array(raw_samples).astype(np.float32)
        float_samples /= np.iinfo(raw_samples.typecode).max  # 归一化到 [-1, 1]

        audio_msg = Float32MultiArray()
        audio_msg.data = float_samples.tolist()

        pub.publish(audio_msg)
        rospy.loginfo("Published audio data to tts_audio_output")

    except Exception as e:
        rospy.logerr("TTS processing failed: %s", str(e))

def tts_node():
    """
    ROS 节点主函数，初始化节点，订阅话题，发布话题。
    """
    global pub
    rospy.init_node('tts_node', anonymous=True)

    rospy.Subscriber("tts_input", String, tts_callback)
    pub = rospy.Publisher('tts_audio_output', Float32MultiArray, queue_size=10) # queue_size 根据需求调整

    rospy.loginfo("TTS node started, waiting for text input on topic 'tts_input'")
    rospy.spin()

if __name__ == '__main__':
    try:
        tts_node()
    except rospy.ROSInterruptException:
        pass