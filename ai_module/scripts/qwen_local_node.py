#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import rospy
from sensor_msgs.msg import CompressedImage  # Import CompressedImage message type
from std_msgs.msg import String
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from ai_module.srv import ProcessImage, ProcessImageResponse, ProcessImageRequest
import base64  # Import the base64 library
import actionlib
from ai_module.msg import MoveBaseAction, MoveBaseGoal

prompt = """你是一个机器人，正在通过相机观察周围环境。你的任务是 识别并描述 你所看到的场景，重点关注 [障碍物、可行走区域、以及其他移动物体]。基于你的场景理解，决定你的下一步动作 (从 '前进', '后退', '左转', '右转', '停止' 中选择一个)，全面探索世界。\n
具体固定输出格式例子如下：\n
{"scene_description": "[口语化对场景的文字描述，**重点描述障碍物、可行走区域和移动物体**]","next_action": "[**从 '前进', '后退', '左转', '右转', '停止' 中选择一个动作**]"}
"""

class QwenVLNode:
    def __init__(self):
        rospy.init_node('qwen_vl_node', anonymous=True)

        # ROS parameters
        self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_raw/compressed") # Image topic from parameter
        # self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_rect_color/compressed") 
        self.qwen_vl_response_topic = rospy.get_param("~qwen_vl_response_topic", "/tts_input") # Output topic from parameter
        self.model_path = rospy.get_param('~model_path', "/home/zhaohang/Qwen2.5-VL-3B-Instruct/model/Qwen2.5-VL-3B-Instruct") # Model path parameter
        self.prompt_prefix = rospy.get_param('~prompt_prefix', prompt) # Default prompt prefix
        self.current_prompt = self.prompt_prefix # Initialize current prompt
        self.image_data = None 

        # Load model and processor from path
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype="auto", device_map="auto"
            ) 
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            rospy.loginfo("Qwen-VL model and processor loaded successfully from: %s", self.model_path)
        except Exception as e:
            rospy.logerr("Failed to load Qwen-VL model or processor from %s: %s", self.model_path, str(e))
            exit(1)

        # ROS Subscribers and Publishers
        self.image_subscriber = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback, queue_size=1) # Image subscriber
        self.qwen_vl_response_publisher = rospy.Publisher(self.qwen_vl_response_topic, String, queue_size=1) # Response publisher

        # ROS Service Server
        self.process_image_service = rospy.Service(
            'process_image', ProcessImage, self.process_image_callback # Service for manual processing
        )
       
        # ROS Action Client
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction) #  定义为 self.client
        rospy.loginfo("Waiting for action server to start...")
        self.client.wait_for_server()
        rospy.loginfo("Action server started, client initialized.")

        rospy.loginfo("Qwen Local Node 初始化成功!")

    def image_callback(self, msg):
        """Callback for compressed image topic."""
        try:
            compressed_data = msg.data
            base64_encoded_bytes = base64.b64encode(compressed_data)
            base64_string = base64_encoded_bytes.decode('utf-8')
            self.image_data = base64_string
        except Exception as e:
            rospy.logerr("Base64 encoding error in image_callback: %s", e)
            self.image_data = None

    def process_image_callback(self, req):
        """Callback for process_image service, now receives prompt in request."""
        rospy.loginfo("Process image service called.")
        if self.image_data is None:
            rospy.logwarn("No image received yet, cannot process.")
            return ProcessImageResponse()

        prompt_text = req.prompt # Get prompt from service request

        if "自由探索" in prompt_text:
            rospy.loginfo("进入自由探索模式...")
            # self.qwen_vl_response_publisher.publish("进入自由探索模式") # 发布场景描述
            self.call_free_explore() # 调用自由探索处理函数
        else:
            rospy.loginfo("进入正常Prompt处理模式...")
            response_msg = self.call_qwen_vl_and_publish_response(self.image_data, prompt_text) # 获取场景描述，不发送action
            self.qwen_vl_response_publisher.publish(response_msg) # 发布场景描述
            rospy.loginfo("Published description to topic: %s", self.qwen_vl_response_topic)
            # self.current_prompt = self.prompt_prefix # No longer need to reset default prompt here, prompt is from service now
        return ProcessImageResponse()

    def call_qwen_vl_and_publish_response(self, image_data, prompt):
        """Calls Qwen-VL model, publishes response to ROS topic and returns the text response."""
        full_response_text = ""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                         "type": "image", 
                         "image": "data:image;base64,"+image_data
                        }, # Use processed image data
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0] # Get the first and only output
            
            full_response_text += output_text
            rospy.loginfo("Qwen-VL Generated description: %s", full_response_text)
        except Exception as e:
            rospy.logerr("Error during image processing or inference: %s", str(e))
            full_response_text = "" # Indicate error in scene description

        return full_response_text # Return only scene description
    
    def call_free_explore(self):
        """专门处理自由探索模式的函数"""
        try:
            response_text = self.call_qwen_vl_and_publish_response(self.image_data, self.prompt_prefix)
            json_response = json.loads(response_text)
            scene_description = json_response.get("scene_description", "") # 获取场景描述
            self.qwen_vl_response_publisher.publish(scene_description) # 发布场景描述
            rospy.loginfo("Published description to topic: %s", self.qwen_vl_response_topic)
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
            # self.call_free_explore() # 调用自由探索处理函数
            process_image_client = rospy.ServiceProxy('process_image', ProcessImage) # 创建服务客户端
            # 创建服务请求对象
            request = ProcessImageRequest()
            request.prompt = "自由探索"  # 设置请求的 prompt 字段为 "自由探索"
            response = process_image_client(request) # 调用服务并获取响应
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call service process_image: %s", e)


    def run(self):
        """Runs the node - not needed for callback-based node."""
        rospy.spin() # Keep node running for callbacks


if __name__ == '__main__':
    try:
        qwen_vl_node = QwenVLNode()
        qwen_vl_node.run() # Use run method to spin the node
    except rospy.ROSInterruptException:
        pass