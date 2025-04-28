#!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import CompressedImage  # 修改为 CompressedImage 消息类型
from std_msgs.msg import String
import requests
import json
import base64
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ai_module.srv import ProcessImage, ProcessImageResponse
import actionlib
from ai_module.msg import MoveBaseAction, MoveBaseGoal

prompt = """你是一个机器人，正在通过相机观察周围环境。你的任务是 识别并描述 你所看到的场景，重点关注 [障碍物、可行走区域、以及其他移动物体]。基于你的场景理解，决定你的下一步动作 (从 '前进', '后退', '左转', '右转', '停止' 中选择一个)，避免与障碍物距离过近。\n
用中文回答，具体固定输出格式如下：\n
{"scene_description": "[口语化对场景的文字描述，**重点描述障碍物、可行走区域和移动物体**]","next_action": "[**从 '前进', '后退', '左转', '右转', '停止' 中选择一个动作**]"}
"""

class LlavaProcessorNode:
    def __init__(self):
        rospy.init_node('llava_node', anonymous=True)

        # ROS 参数
        self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_raw/compressed") 
        # self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_rect_color/compressed") 
        self.llava_response_topic = rospy.get_param("~llava_response_topic", "/tts_input")
        self.ollama_api_url = rospy.get_param("~ollama_api_url", "http://localhost:11434/api/generate")
        self.model_name = rospy.get_param("~model_name", "llava")
        self.prompt_prefix = rospy.get_param("~prompt_prefix", prompt)
        self.process_interval = rospy.get_param("~process_interval", 1.0) 
        self.current_prompt = self.prompt_prefix  # 初始化为默认提示词

        # CvBridge
        self.bridge = CvBridge()

        # ROS 发布者和订阅者
        self.image_subscriber = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback, queue_size=1) # 修改为 Image 消息类型
        self.llava_response_publisher = rospy.Publisher(self.llava_response_topic, String, queue_size=1)

        # 创建服务服务器
        self.process_image_service = rospy.Service(
            'process_image', ProcessImage, self.process_image_callback
        )

        # ROS Action Client
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction) #  定义为 self.client
        rospy.loginfo("Waiting for action server to start...")
        self.client.wait_for_server()
        rospy.loginfo("Action server started, client initialized.")

        rospy.loginfo("LLaVA Processor Node 初始化成功!")

    def encode_cv_image_to_base64(self, cv_image):
        """
        将 OpenCV 图像编码为 Base64 字符串。
        """
        _, image_buffer = cv2.imencode('.jpg', cv_image)
        encoded_string = base64.b64encode(image_buffer).decode('utf-8')
        return encoded_string

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
            return ProcessImageResponse()
        prompt_text = req.prompt

        if "自由探索" in prompt_text:
            rospy.loginfo("进入自由探索模式...")
            self.llava_response_publisher.publish("进入自由探索模式") # 发布场景描述
            self.call_free_explore() # 调用自由探索处理函数进入自由探索模式
        else:
            rospy.loginfo("进入正常Prompt处理模式...")
            response_msg = self.call_llava_api_and_publish_response(self.image_base64, prompt_text) # 获取场景描述，不发送action
            self.llava_response_publisher.publish(response_msg) # 发布场景描述
            rospy.loginfo("Published description to topic: %s", self.llava_response_topic)
            # self.current_prompt = self.prompt_prefix # No longer need to reset default prompt here, prompt is from service now
        return ProcessImageResponse()

    def call_llava_api_and_publish_response(self, image_base64, prompt):
        """
        调用 Ollama API 发送图像和文本提示给 LLaVA 模型，并获取回复，然后发布到 ROS 话题。
        """
        full_response_text = ""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_base64]
            }

            headers = {'Content-Type': 'application/json'}

            response = requests.post(self.ollama_api_url, headers=headers, json=payload, stream=True)
            response.raise_for_status()

            rospy.loginfo("LLaVA 模型回复:")
            for line in response.iter_lines():
                if line:
                    json_data = json.loads(line)
                    if 'response' in json_data:
                        response_part = json_data['response']
                        # rospy.loginfo(response_part)
                        print(response_part, end="", flush=True) # 逐字打印
                        full_response_text += response_part
                    if json_data.get('done'):
                        rospy.loginfo("LLaVA response finished.")
                        break

        except requests.exceptions.RequestException as e:
            rospy.logerr("API 请求错误: %s", e)
        except json.JSONDecodeError:
            rospy.logerr("错误: API 响应 JSON 解析失败。请检查 Ollama 服务是否正常运行。")
        except Exception as e:
            rospy.logerr("发生未知错误: %s", e)
        
        return full_response_text

    def call_free_explore(self):
        """专门处理自由探索模式的函数"""
        try:
            response_text = self.call_llava_api_and_publish_response(self.image_base64, self.prompt_prefix)
            json_response = json.loads(response_text)
            scene_description = json_response.get("scene_description", "") # 获取场景描述
            self.llava_response_publisher.publish(scene_description) # 发布场景描述
            rospy.loginfo("Published description to topic: %s", self.llava_response_publisher)
            try:
                next_action = json_response.get("next_action")
                if next_action:
                    self.send_move_command(next_action,done_cb=self._move_base_done_cb) # 发送 action goal
                else:
                    rospy.logwarn("自由探索模式: Qwen-VL response did not contain 'next_action'. Not sending move command.")
            except json.JSONDecodeError:
                rospy.logwarn("自由探索模式: Could not parse Qwen-VL output as JSON, not sending move command.")

            rospy.loginfo("自由探索模式: 指令已处理，等待Action执行结果...")

        except Exception as e:
            rospy.logerr("自由探索模式处理出错: %s", str(e))
    
    def send_move_command(self, command, done_cb=None): # 添加 done_cb 参数
        """
        发送移动指令到 Action Server 并等待结果，可以传入 done_cb.
        """
        self.goal = MoveBaseGoal()
        self.goal.command = command

        rospy.loginfo("Sending goal: %s", command)
        if done_cb is not None:
            self.client.send_goal(self.goal, done_cb=done_cb) # 如果提供了 done_cb，则使用
        else:
            self.client.send_goal(self.goal) 

    def _move_base_done_cb(self, goal_status, goal_result): #  Action 完成回调函数
        """MoveBase Action 完成后的回调函数，用于自由探索模式"""
        rospy.loginfo("自由探索模式: Action执行完毕，准备下一次图像处理...")
       # 使用 rospy.Timer 延迟 1 秒 (或你需要的延迟时间) 后调用 self.call_free_explore
        rospy.Timer(rospy.Duration(5), self._delayed_free_explore_callback, oneshot=True) # oneshot=True 表示定时器只触发一次

    def _delayed_free_explore_callback(self, event): # 定时器回调函数
        """延迟调用 self.call_free_explore() 的回调函数"""
        try:
            self.call_free_explore() # 调用自由探索处理函数
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call service process_image: %s", e)
        
    def run(self):
        """Runs the node - not needed for callback-based node."""
        rospy.spin() # Keep node running for callbacks

if __name__ == '__main__':
    try:
        llava_processor_node = LlavaProcessorNode()
        llava_processor_node.run()
    except rospy.ROSInterruptException:
        pass