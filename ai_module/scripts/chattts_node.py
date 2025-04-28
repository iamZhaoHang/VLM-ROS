#!/usr/bin/env python

import rospy
from std_msgs.msg import String, ByteMultiArray # 订阅文本回复 String, 发布音频 ByteMultiArray
import ChatTTS  # 导入 ChatTTS 库
import torch
import torchaudio
import gc
import numpy as np
import tempfile # 用于创建临时文件

#此文件当时仅用来测试ChatTTS文本转语音模型的效果，经过测试发现处理时长太长，遂放弃此文件。

MODEL_PATH = '/home/zhaohang/ChatTTS/ChatTTS-Model'

class ChatttsProcessorNode:
    def __init__(self):
        rospy.init_node('chattts_node', anonymous=True)

        # ROS 参数
        self.llava_response_topic = rospy.get_param("~llava_response_topic", "/tts_input") # 订阅文本回复的话题
        self.audio_topic_name = rospy.get_param("~audio_topic", "/tts_audio_output")

        # 初始化 ChatTTS 模型
        self.chattts_model = ChatTTS.Chat()
        self.chattts_model.load(
            source="local",
            custom_path=MODEL_PATH,
            # device=torch.device("cpu"), # 使用 CPU
            compile=False,
        )

        # ROS 发布者和订阅者
        self.llava_response_subscriber = rospy.Subscriber(self.llava_response_topic, String, self.llava_response_callback, queue_size=1) # 订阅文本回复
        self.audio_publisher = rospy.Publisher(self.audio_topic_name, ByteMultiArray, queue_size=1)

        rospy.loginfo("ChatTTS Node initialized.")

    def text_to_speech_and_publish(self, text_response):
        """
        使用 ChatTTS 将文本转换为语音并发布为 ROS 话题。
        """
        try:
            wavs = self.chattts_model.infer(text_response, use_decoder=True) # 使用 ChatTTS 推理文本

            for i in range(len(wavs)):
                # 保存音频文件到本地文件（采样率为24000Hz）
                output_file = f"basic_output{i}.wav"
                torchaudio.save(output_file, torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
                print(f"音频文件已保存: {output_file}")

            if wavs and len(wavs) > 0: # 确保 wavs 列表不为空
                # 将音频数据转换为 ByteMultiArray 消息
                audio_data = np.int16(wavs[0] * 32767).tobytes() # 缩放并转换为 int16，然后转为字节
                audio_msg = ByteMultiArray(data=audio_data)
                self.audio_publisher.publish(audio_msg)
                rospy.loginfo("ChatTTS audio published to topic: %s", self.audio_topic_name)

            else:
                rospy.logwarn("ChatTTS 未生成有效的音频数据。")

        except Exception as e:
            rospy.logerr("ChatTTS 文本转语音过程中出错: %s", e)

    def llava_response_callback(self, msg):
        """
        LLaVA 文本回复话题的回调函数。
        """
        llava_response_text = msg.data # 从 String 消息中获取文本数据
        rospy.loginfo("Received LLaVA response text: %s", llava_response_text)
        self.text_to_speech_and_publish(llava_response_text)


if __name__ == '__main__':
    try:
        chattts_processor_node = ChatttsProcessorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass