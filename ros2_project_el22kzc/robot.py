import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from math import sin, cos
from geometry_msgs.msg import PoseStamped

class Robot(Node):
    def __init__(self):
        super().__init__('robot')
       
        # Initialise a publisher to publish messages to the robot base
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rate = self.create_rate(10)  # 10 Hz

        # Flags for movement and color detection
        self.moveForwardsFlag = False
        self.moveBackwardsFlag = False
        
        #Colour Flag
        self.myColourFlag = False
        self.greenDetected = False
        self.redDetected = False

        # Sensitivity for color detection
        self.sensitivity = 10

        # Initialise CvBridge() for image conversion
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.subscription  # prevent unused variable warning
        
        #Initially velocity, And parameters for turning
        self.x_velocity = 0.2
        self.move_slow = False
        self.angular_turn = 0
        
        #Map
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.using_nav2 = True  # Start in Nav2 mode
        
        #Tracking goal
        self.goal_index = 0
        self.nav_goal_done = True
        self.goal_list = [(0.0,-5.0,0.0),(0.0,-9.0,0.0),(0.0,-10.0,0.0),(-6.0,-10.0,0.0),(-7.0,-4.0,0.0)]  # Example waypoints
        
        #For remembering blue goal
        self.goal_position = [0,0]
        self.goal_orientation = [0,0]
        self.found_goal = False
        self.return_goal = False
        


    #Real time feedback
    def callback(self, data):
        # Convert the received image into an OpenCV image
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        cv2.namedWindow('camera_Feed',cv2.WINDOW_NORMAL)
        
        cv2.resizeWindow('camera_Feed',320,240)
        cv2.waitKey(3)


        # Convert image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Set color ranges for green detection
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        
        # Set color ranges for blue detection
        hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])
        
        # Set color ranges for red detection Start edge
        hsv_red1_lower = np.array([0 - self.sensitivity, 100, 100])
        hsv_red1_upper = np.array([0 + self.sensitivity, 255, 255])
        # Set color ranges for red detection End edge
        hsv_red2_lower = np.array([180 - self.sensitivity, 100, 100])
        hsv_red2_upper = np.array([180 + self.sensitivity, 255, 255])
        
        
        # Create mask to isolate green color
        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        
        # Red mask
        red_mask_start = cv2.inRange(hsv_image, hsv_red1_lower, hsv_red1_upper)
        red_mask_end = cv2.inRange(hsv_image, hsv_red2_lower, hsv_red2_upper)
        red_mask =  cv2.bitwise_or(red_mask_start, red_mask_end)
      
        # Blue mask
        blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)

        # Find contours in the mask
        green_contours, _ = cv2.findContours(green_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        all_contours = len(green_contours)+len(red_contours)+len(blue_contours) #only for other colour movement detection

        # Check for green object and set flags based on size
        if all_contours > 0:
            #self.using_nav2 = False;
            #largest_contour = max(contours, key=cv2.contourArea)
            #contour_area = cv2.contourArea(largest_contour)
            
            #Detect red
            if(len(red_contours)>0):
                red_con = max(red_contours, key=cv2.contourArea)
                red_M = cv2.moments(red_con)
                if red_M['m00'] != 0:#This is to prevent divison by zero
                        cx, cy = int(red_M['m10']/red_M['m00']), int(red_M['m01']/red_M['m00'])
            
                        #Draw red
                        if cv2.contourArea(red_con) > 10:
                                x, y, width, height = cv2.boundingRect(red_con)
                                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 3)  # Red rectangle
                                cv2.putText(image, 'Red', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX,0.6, (0, 255, 0), 3)
                                self.redDetected = True#Show red detected
   
            #Detect blue
            if(len(blue_contours)>0):
                blue_con = max(blue_contours, key=cv2.contourArea)
                blue_M = cv2.moments(blue_con)
                self.myColourFlag = True
                if blue_M['m00'] != 0:#This is to prevent divison by zero
                        cx, cy = int(blue_M['m10']/blue_M['m00']), int(blue_M['m01']/blue_M['m00'])
                
                        
                        blue_area = cv2.contourArea(blue_con)
                
                        #print(blue_area)
                        #print(image.shape[1],image.shape[1])
                        #print("Area : "+str(blue_area)+", H: "+str(image.shape[0])+", W: "+str(image.shape[1]))
                        
                        camera_center_x = image.shape[1] // 2 #Get camera mid-point ,width / 2 
                        direction_turn = camera_center_x-cx
                
                        factor_turn = 0.002#Sensitivty
                        self.angular_turn = factor_turn*direction_turn
                
                        #Draw blue
                        if blue_area > 10:
                                x, y, width, height = cv2.boundingRect(blue_con)
                                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 3)  # Blue rectangle
                                cv2.putText(image, 'Blue', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX,0.6, (0, 255, 0), 3)
                        
                        if blue_area > 1000:
                                #print("Area : "+str(blue_area)+", H: "+str(h)+", W: "+str(w))
                        
                                if blue_area > 330000:  # Move backward if too big
                                        self.myColourFlag = True
                                        self.moveForwardsFlag = False
                                        self.moveBackwardsFlag = True
                                        self.move_slow = False

                                elif 290000 < blue_area <= 330000:  # Stop at certain distance      
                                        self.myColourFlag = True
                                        self.moveForwardsFlag = False
                                        self.moveBackwardsFlag = False
                                        self.move_slow = True

                                else:  # Move forward
                                        self.myColourFlag = True
                                        self.moveForwardsFlag = True
                                        self.moveBackwardsFlag = False
                                        self.move_slow = False


                        
            
            #Detect green
            if(len(green_contours)>0):
                green_con = max(green_contours, key=cv2.contourArea)
                green_M = cv2.moments(green_con)
                if green_M['m00'] != 0:#This is to prevent divison by zero
                        cx, cy = int(green_M['m10']/green_M['m00']), int(green_M['m01']/green_M['m00'])
            
                        if cv2.contourArea(green_con) > 10:
                                x, y, width, height = cv2.boundingRect(green_con)
                                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 3)  # Blue rectangle
                                cv2.putText(image, 'Green', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX,0.6, (0, 255, 0), 3)
                                self.greenDetected = True#Show green detected
            
            
            #cv2.imshow("Green Mask", green_mask)
            #cv2.imshow("Red Mask", red_mask)
            #cv2.imshow("Blue Mask", blue_mask)
            

        else:
            self.using_nav2 = True
            self.myColourFlag = False
            self.moveForwardsFlag = False
            self.moveBackwardsFlag = False
       
        cv2.imshow('camera_Feed', image)

    def walk_forward(self):
        # Move forward
        desired_velocity = Twist()
        desired_velocity.linear.x = self.x_velocity
        desired_velocity.angular.z = self.angular_turn
        self.publisher.publish(desired_velocity)

    def walk_backward(self):
        # Move backward
        desired_velocity = Twist()
        desired_velocity.linear.x = self.x_velocity*-1
        self.publisher.publish(desired_velocity)

    def stop(self):
        # Stop the robot
        desired_velocity = Twist()
        self.publisher.publish(desired_velocity)


    ##All method came from the GoToPose class from lab 4

    #Map 
    #Modify the function is use internal coordinate if goal_found
    #Works for storing temporal coordinate, instead of rewriting or creating new function
    #The orientation can be stored but doesnt seem to work,still I left the previous code intact
    def send_goal(self, x, y, yaw,goal_found=False):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Orientation
        if goal_found:
                goal_msg.pose.pose.orientation.z = float(self.goal_orientation[0])
                goal_msg.pose.pose.orientation.w = float(self.goal_orientation[1])
        else:
                goal_msg.pose.pose.orientation.z = sin(yaw / 2)
                goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')
        self.nav_goal_done = True

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # NOTE: if you want, you can use the feedback while the robot is moving.
        #       uncomment to suit your need.

        if(not self.found_goal and self.myColourFlag):
        ## Access the current pose
                current_pose = feedback_msg.feedback.current_pose
                self.goal_position[0] = current_pose.pose.position.x
                self.goal_position[1] = current_pose.pose.position.y
                self.goal_orientation[0] = current_pose.pose.orientation.z
                self.goal_orientation[1] = current_pose.pose.orientation.w
                #Prevent from being called again
                self.found_goal = True

        ## Access other feedback fields
        #navigation_time = feedback_msg.feedback.navigation_time
        #distance_remaining = feedback_msg.feedback.distance_remaining

        ## Print or process the feedback data
        #self.get_logger().info(f'Current Pose: [x: {position.x}, y: {position.y}, z: {position.z}]')
        #self.get_logger().info(f'Distance Remaining: {distance_remaining}')






def main():
    def signal_handler(sig, frame):
        robot.stop()
        rclpy.shutdown()

    # Instantiate the robot node
    rclpy.init(args=None)
    robot = Robot()

    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            print("Green Detected: "+str(robot.greenDetected))
            print("Red Detected: "+str(robot.redDetected))
            print("Goal found: "+str(robot.found_goal))

            #Ensure robot must detect all colour before heading to goal
            if(robot.greenDetected and robot.redDetected and robot.found_goal):
                if(robot.return_goal):
                        #Colour movement tracking once finished detection
                        #Will rotate 360 to find blue object if not found
                        if(robot.myColourFlag):
                                
                                #Act as dampener to prevent overshoot feedback
                                if robot.move_slow:
                                        robot.x_velocity = 0.01
                                else:
                                        robot.x_velocity = 0.2
            
                                if robot.moveForwardsFlag:
                                        #print("Moving Forward")
                                        robot.walk_forward()
                                elif robot.moveBackwardsFlag:
                                        robot.walk_backward()
                                        #print("Moving Backward")
                                else:	
                                        #print("Stopping")
                                        robot.stop()
                        else:
                                #Orientation cant seem to be saved even after saving it,however position is saved
                                #Therefore will rotate 360 to find blue
                                robot.angular_turn = 0.2
                                robot.x_velocity = 0.01
                                robot.walk_forward()
                                robot.angular_turn = 0.0
                                
                                
                else:
                        if(robot.nav_goal_done):
                                robot.nav_goal_done = False
                                print("Running set goal")
                                robot.send_goal(float(robot.goal_position[0]),float(robot.goal_position[1]),0.0,True)
                                robot.return_goal = True

            else:
                #If all colour havent been detected will continue on path until the goal list ran out
                if robot.using_nav2 and robot.nav_goal_done:
                        print("On Goal")
                        if robot.goal_index < len(robot.goal_list): #Prevent out of range index error
                                print("Goal: "+str(robot.goal_index))
                                x, y, z = robot.goal_list[robot.goal_index]  
                                robot.send_goal(x,y,z)
                                robot.nav_goal_done = False
                                robot.goal_index = robot.goal_index + 1
                        
            robot.rate.sleep()

    except ROSInterruptException:
        pass

    # Clean up
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

