import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

model_path = 'Models/gesture_recognizer.task'

class GestureControlNode(Node):

    def __init__(self):
        super().__init__('gesture_control_node')
        self.publisher = self.create_publisher(Twist, '/turtlesim/turtle1/cmd_vel', 10)
        
    def send_command(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.publisher.publish(msg)

gesture_msg = ""
current_gesture = None

def print_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global gesture_msg, current_gesture
    node = GestureControlNode()
    if result.gestures:
        for gesture in result.gestures:
            current_gesture = gesture[0].category_name
            gesture_msg = f'Gesture: {gesture[0].category_name}, Score: {gesture[0].score:.2f}'

            if current_gesture == 'Thumb_Up':
                node.send_command(1.0, 0.0)  # Move forward
            elif current_gesture == 'Closed_Fist':
                node.send_command(0.0, 0.0)  # Stop
            elif current_gesture == 'Pointing_Up':
                node.send_command(0.0, 1.0)  # Turn left
            elif current_gesture == 'Victory':
                node.send_command(0.0, -1.0)  # Turn right
    else:
        gesture_msg = 'No gestures detected.'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

def main(args=None):
    rclpy.init(args=args)
    
    node = GestureControlNode()

    cap = cv2.VideoCapture(0)

    with GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            recognizer.recognize_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

            if current_gesture:
                cv2.putText(frame, current_gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            rclpy.spin_once(node, timeout_sec=0.1)

    cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
