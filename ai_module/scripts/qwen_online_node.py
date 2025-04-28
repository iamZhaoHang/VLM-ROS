#!/usr/bin/python

import rospy
from sensor_msgs.msg import CompressedImage  # 修改为 CompressedImage 消息类型
from std_msgs.msg import String
import requests  # 移除 requests
import json
import base64
import torch
import gc
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ai_module.srv import ProcessImage, ProcessImageResponse
import os  # 导入 os 模块
from openai import OpenAI  # 导入 openai 库

class LlavaProcessorNode:
    def __init__(self):
        rospy.init_node('llava_node', anonymous=True)

        # ROS 参数
        self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_raw/compressed")
        # self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_rect_color/compressed")
        self.llava_response_topic = rospy.get_param("~llava_response_topic", "/tts_input")
        self.dashscope_api_key = rospy.get_param("~dashscope_api_key", "sk-##################################") #替换为自己的key
        self.model_name = rospy.get_param("~model_name", "qwen2.5-vl-7b-instruct") 
        self.prompt_prefix = rospy.get_param("~prompt_prefix", "假设你是一个机器人，你看到的图像来自你的视觉传感器，请用中文回答，你看见了什么？；并决定你的下一步动作，如 前进 左转等")
        self.process_interval = rospy.get_param("~process_interval", 1.0)
        self.dashscope_base_url = rospy.get_param("~dashscope_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1") # 添加 DashScope API base URL

        self.prompt_topic = rospy.get_param("~prompt_topic", "/llava_prompt")
        self.current_prompt = self.prompt_prefix  # 初始化为默认提示词

        # CvBridge
        self.bridge = CvBridge()

        # ROS 发布者和订阅者
        self.image_subscriber = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback, queue_size=10) # 修改为 Image 消息类型
        self.llava_response_publisher = rospy.Publisher(self.llava_response_topic, String, queue_size=1)
        self.prompt_subscriber = rospy.Subscriber(self.prompt_topic, String, self.prompt_callback)

        # 创建服务服务器
        self.process_image_service = rospy.Service(
            'process_image', ProcessImage, self.process_image_callback
        )

        self.last_process_time = 0.0  # 初始化上次处理时间
        self.image_base64 = None # 初始化 image_base64
        self.openai_client = OpenAI(  # 初始化 OpenAI 客户端
            api_key=self.dashscope_api_key,
            base_url=self.dashscope_base_url,
        )
        rospy.loginfo("LLaVA Processor Node initialized.")

    def prompt_callback(self, msg):
        self.current_prompt = msg.data
        rospy.loginfo("Received new prompt: %s", msg.data)

    def encode_cv_image_to_base64(self, cv_image):
        """
        将 OpenCV 图像编码为 Base64 字符串。
        """
        _, image_buffer = cv2.imencode('.jpg', cv_image)
        encoded_string = base64.b64encode(image_buffer).decode('utf-8')
        return encoded_string

    def call_llava_api_and_publish_response(self, image_base64, prompt):
        """
        调用 DashScope API 发送图像和文本提示给 qwen 模型，并获取回复，然后发布到 ROS 话题。
        """
        full_response_text = ""
        try:
            rospy.loginfo("Calling DashScope API...")
            completion = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user","content": [
                        {"type": "text","text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}} # 使用 base64 编码的图像数据
                        ]}]
                )

            rospy.loginfo("DashScope 模型回复:")
            response_text = completion.choices[0].message.content # 获取模型回复文本
            print(response_text) # 打印完整回复
            full_response_text = response_text


            # 在 LLaVA 回复结束后，尝试清空 GPU 缓存 (DashScope 在云端运行，这里不需要清空本地 GPU 缓存)
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     gc.collect()
            #     rospy.loginfo("LLaVA 回复结束，已清空 GPU 缓存并进行垃圾回收。")
            # else:
            #     rospy.loginfo("CUDA 不可用，跳过 GPU 缓存清理。")

            # 发布 LLaVA 文本回复到 ROS 话题
            response_msg = String(data=full_response_text)
            self.llava_response_publisher.publish(response_msg)
            rospy.loginfo("LLaVA response text published to topic: %s", self.llava_response_topic)


        except openai.APIError as e: # 捕获 openai 库的 APIError 异常
            rospy.logerr("API 请求错误: %s", e)
        except Exception as e:
            rospy.logerr("发生未知错误: %s", e)

    def image_callback(self, msg):
        """
        图像话题的回调函数。
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.image_base64 = self.encode_cv_image_to_base64(cv_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge 错误: %s", e)
        except Exception as e:
            rospy.logerr("图像回调函数出错: %s", e)

    def process_image_callback(self, req):
        rospy.loginfo("Process image service called.")
        if self.image_base64 is None:
            rospy.logwarn("No image received yet, cannot process.")
            return ProcessImageResponse(response_text="") # 返回空字符串表示没有图像处理
        try:
            prompt_text = self.current_prompt
            self.call_llava_api_and_publish_response(self.image_base64, prompt_text)
            return ProcessImageResponse(response_text="Processing image...") # 服务调用成功，返回正在处理的信息
        except Exception as e:
            rospy.logerr("Error in process_image_callback: %s", e)
            return ProcessImageResponse(response_text="") # 发生错误时返回空字符串


if __name__ == '__main__':
    try:
        llava_processor_node = LlavaProcessorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass