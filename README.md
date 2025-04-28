# VLM-ROS
为了实现真正的All in Local！ 我将Llava视觉大模型、QWen2.5-VL多模态大模型，以及STT和TTS模型全部部署在本地计算机上，打造了一个完全离线的机器人视觉交互系统。 机器人通过摄像头感知周围环境，LLaVA和QWen2.5-VL进行视觉分析，STT进行语音识别，TTS进行语音播报，整个过程完全在本地完成。

1.准备工作
    a.本地部署Qwen 2.5 - VL，参考官网教程：https://github.com/QwenLM/Qwen2.5-VL/tree/main
    b.本地部署sherpa-onnx框架，参考官网教程：https://github.com/k2-fsa/sherpa-onnx

2.功能包用途
    (1) ai_module:
        scripts文件夹下，包含VLM处理rospy节点
        src文件夹下，包含自定义Action Server
    (2) sherpa_onnx_ros:
        包含TTS、ASR语音处理功能
    (3) sound_service：
        将此功能包部署在机器人端侧，负责采集音频和播放音频