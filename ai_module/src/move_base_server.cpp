#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <ai_module/MoveBaseAction.h>
#include <geometry_msgs/Twist.h>

typedef actionlib::SimpleActionServer<ai_module::MoveBaseAction> Server;

class MoveBaseActionServer
{
private:
  ros::NodeHandle nh_;
  Server server_;
  std::string action_name_;
  geometry_msgs::Twist cmd_vel_msg_;
  ros::Publisher cmd_vel_pub_;

public:
  MoveBaseActionServer(std::string name) :
    server_(nh_, name, boost::bind(&MoveBaseActionServer::executeCB, this, _1), false),
    action_name_(name)
  {
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    server_.start();
    ROS_INFO("Action server [%s] started!", action_name_.c_str());
  }

  ~MoveBaseActionServer(void)
  {
  }

  void executeCB(const ai_module::MoveBaseGoalConstPtr &goal)
  {
    ROS_INFO_STREAM("Action server: Received goal command: " << goal->command);
    ai_module::MoveBaseResult result_;
    bool success = true;

    ROS_DEBUG("Action server: executeCB started for command: %s", goal->command.c_str()); // **Added: Entry point log**

    // 根据接收到的指令执行不同的动作
    if (goal->command == "前进") {
        ROS_INFO("Executing command: 前进");
        cmd_vel_msg_.linear.x = 0.3; // 设置线速度
        cmd_vel_msg_.angular.z = 0.0;
    } else if (goal->command == "后退") {
        ROS_INFO("Executing command: 后退");
        cmd_vel_msg_.linear.x = -0.2;
        cmd_vel_msg_.angular.z = 0.0;
    } else if (goal->command == "左转") {
        ROS_INFO("Executing command: 左转");
        cmd_vel_msg_.linear.x = 0.3;
        cmd_vel_msg_.angular.z = 0.3; // 设置角速度
    } else if (goal->command == "右转") {
        ROS_INFO("Executing command: 右转");
        cmd_vel_msg_.linear.x = 0.3;
        cmd_vel_msg_.angular.z = -0.3;
    } else if (goal->command == "停止") {
        ROS_INFO("Executing command: 停止");
        cmd_vel_msg_.linear.x = 0.0;
        cmd_vel_msg_.angular.z = 0.0;
    } else {
        ROS_WARN_STREAM("Unknown command: " << goal->command);
        result_.success = false;
        result_.message = "Unknown command received";
        ROS_DEBUG("Action server: Setting state to ABORTED for unknown command: %s", goal->command.c_str()); // **Added: Debug before setAborted**
        server_.setAborted(result_); // 设置 Action 为 Aborted 状态
        ROS_DEBUG("Action server: setAborted() called for unknown command: %s", goal->command.c_str()); // **Added: Debug after setAborted**
        success = false;
    }

    if (success) {
        // 发布 cmd_vel 消息
        cmd_vel_pub_.publish(cmd_vel_msg_);
        ros::Duration(5.0).sleep(); // 假设动作执行 3 秒
        cmd_vel_msg_.linear.x = 0.0; // 停止移动
        cmd_vel_msg_.angular.z = 0.0;
        cmd_vel_pub_.publish(cmd_vel_msg_);

        result_.success = true;
        result_.message = "Command executed successfully";
        server_.setSucceeded(result_); // 设置 Action 为 Succeeded 状态
        ROS_INFO("Action finished: Succeeded");
    }
  }
};

int main(int argc, char** argv)
{
  setlocale(LC_ALL,"");
  ros::init(argc, argv, "move_base_action_server");
  MoveBaseActionServer server("move_base"); // Action 名称为 "move_base"
  ros::spin();
  return 0;
}