#!/usr/bin/python3
import numpy as np
import cv2
import depthai as dai

# import matplotlib.pyplot as plt
import sys

sys.path.append(".")
#import matrix_processing as matrix_processing
#from outlier_filter import *

pp = dai.Pipeline()
#imu = pp.create(dai.node.IMU)
# create monocams because stereo wants them
left_lens = pp.createMonoCamera()
right_lens = pp.createMonoCamera()
# ctrlIn = pp.create(dai.node.XLinkIn)
# ctrlIn.setStreamName("control")
# set resolution to 400p to save space
left_lens.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right_lens.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left_lens.setFps(100)
right_lens.setFps(100)
# create stereo depth
stereo = pp.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)


rgbCamSocket = dai.CameraBoardSocket.CAM_B


stereo.initialConfig.setConfidenceThreshold(250)
stereo.setRectifyEdgeFillColor(0)
stereo.setLeftRightCheck(False)
stereo.setDepthAlign(rgbCamSocket)
#xoutImu = pp.createXLinkOut()
xoutDepth = pp.createXLinkOut()
xoutLeft = pp.createXLinkOut()
xoutRight = pp.createXLinkOut()

# ctrlIn.out.link(left_lens.inputControl)
# ctrlIn.out.link(right_lens.inputControl)

#xoutImu.setStreamName("imu")
xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")
xoutDepth.setStreamName("depth")

# some stereo config stuff


#imu_freq = 5
# imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, imu_freq)
#imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, imu_freq)
# imu.enableIMUSensor(dai.IMUSensor.MAGNETOMETER_RAW, imu_q_size)
#imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, imu_freq)
#imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, imu_freq)


#imu.setBatchReportThreshold(20)
#imu.setMaxBatchReports(20)
# Link plugins IMU -> XLINK
#imu.out.link(xoutImu.input)

#
left_lens.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right_lens.setBoardSocket(dai.CameraBoardSocket.CAM_C)

#
left_lens.out.link(stereo.left)
right_lens.out.link(stereo.right)

#
stereo.disparity.link(xoutDepth.input)
DEPTH=True
DISPLAY =True and DEPTH
if DISPLAY:
    cv2.startWindowThread()
    cv2.namedWindow("scatter")
    #cv2.startWindowThread()
    #cv2.namedWindow("raw")

from collections import deque
frames = deque()
#from outlier_filter import *
from messenger import *
from viz import *
import rospy
import numpy_multiply as cuda_processing
ROS = True
RATE = 25
CUTOFF = 180

import time
def main():
    val0 = cuda_processing.warmup(1)
    dx = 82 * np.pi / 180 / (2 * 310 + 1)
    val = np.zeros(400*640) + 10000
    ages_np = np.zeros(400*640)
    rolling_frame_ptr = cuda_processing.move_to_gpu(val)

    conv_arr1 = np.zeros(400*640)
    conv_arr1_ptr = cuda_processing.move_to_gpu(conv_arr1)

    conv_arr2 = np.zeros(400*640)
    conv_arr2_ptr = cuda_processing.move_to_gpu(conv_arr2)

    display_np = np.zeros(400*640)
    ones = np.ones(640)
    ones_gpu = cuda_processing.move_to_gpu(ones)
    markers = np.zeros(640)
    markers_gpu = cuda_processing.move_to_gpu(markers)
    
    markers_over_time = np.zeros(640)+95
    markers_over_time_gpu = cuda_processing.move_to_gpu(markers_over_time)

    markers_age = np.zeros(640)
    markers_age_gpu = cuda_processing.move_to_gpu(markers_age)

    markers_hold = np.zeros(640)
    markers_hold_gpu = cuda_processing.move_to_gpu(markers_hold)
    marker_arr = [markers_gpu,markers_hold_gpu]
    lpf_np = np.array([0,0,0,1,1,1,1,1,1,0,0,0])
    lpf_np = lpf_np * 1/(lpf_np.shape[0]-5)
    lpf_gpu = cuda_processing.move_to_gpu(lpf_np)
    mark = 0
    marker = lambda x: x + 1 if x < 1 else 0
    flt = np.array([0.000133831,0.00443186,0.0539911,0.241971,0.398943,0.241971,0.0539911,0.00443186,0.000133831])
    flt_ptr = cuda_processing.move_to_gpu(flt)

    mean_flt = np.zeros(9) + 1/9
    mean_flt_ptr = cuda_processing.move_to_gpu(mean_flt)

    deriv_flt = np.array([-1,1])
    deriv_flt_ptr = cuda_processing.move_to_gpu(deriv_flt)

    ages_ptr = cuda_processing.move_to_gpu(ages_np)
    new_vals_ptr = cuda_processing.move_to_gpu(ages_np)
    theta_max = 82 * np.pi / 180
    #dest_ptr = cuda_processing.move_to_gpu_addr(new_vals, val-10000)
    if ROS:
        rospy.init_node("populate_laserscan", anonymous=True)
        # Create a publisher
        laserscan_pub = rospy.Publisher("/scan", LaserScan, queue_size=30)
        imu_pub = rospy.Publisher("/imu_oak", Imu, queue_size=10)

    FIRST = 5
    qdepth = None
    depthFrame = None
    s = time.perf_counter()
    with dai.Device(pp) as device:
        if DEPTH:
            qdepth = device.getOutputQueue(name="depth", maxSize=70, blocking=False)
        #imuQueue = device.getOutputQueue(name="imu", maxSize=10, blocking=False)

        c = 19
        num = 882.5 * 7.5  # focal point * baseline for OAK-D
        #start = time.time()
        while not rospy.is_shutdown() and True:
            # nonblocking try to get frames
            #if DEPTH:
            depthFrame = qdepth.tryGet()
            #imuData = imuQueue.tryGet()
            if False and imuData != None:
                if ROS and True:
                    packets = imuData.packets
                    for packet in packets:
                        # FOR MOCKING:
                        # packet = getMockIMU(packet)
                        imu_msg = populate_imu(packet)
                        imu_pub.publish(imu_msg)

            if DEPTH and depthFrame != None:

                #depthFrame = 
                cuda_processing.move_to_gpu_addr(new_vals_ptr, depthFrame.getFrame(),640*400)
                #cuda_processing.invert_wrap(new_vals_ptr, num, 640, 400)
                # apply minimization
                cuda_processing.direct_minimize_wrap(rolling_frame_ptr, ages_ptr, new_vals_ptr,4, 640,400, 640*400)

                if DISPLAY and not c%5:
                    #s = time.perf_counter()
                    # gaussian blurring
                    #cuda_processing.direct_conv_wrap(rolling_frame_ptr,conv_arr1_ptr,flt_ptr, flt.shape[0], 640, 400,400*640)
                    #cuda_processing.scalar_divide_wrap(conv_arr1_ptr, -1, 640, 400)
                    # apply mean filter
                    cuda_processing.direct_conv_wrap(rolling_frame_ptr, conv_arr2_ptr, mean_flt_ptr, mean_flt.shape[0], 640, 400, 400*640)
                    # apply first derivative
                    cuda_processing.direct_conv_wrap(conv_arr2_ptr, conv_arr1_ptr, deriv_flt_ptr, deriv_flt.shape[0], 640, 400, 400*640)
                    # find markers
                    #cuda_processing.move_to_cpu(conv_arr1_ptr,display_np)

                    cuda_processing.find_min_wrap(conv_arr1_ptr, markers_gpu, rolling_frame_ptr,-0.15 ,80, 200,640, 400)

                    #cuda_processing.age_out_wrap(markers_over_time_gpu, markers_age_gpu, markers_gpu, 2, 20,32,640)
                    #cuda_processing.c_add_wrap(markers_over_time_gpu, ones_gpu,1,640,640)
                    #cuda_processing.scalar_divide_wrap(markers_over_time_gpu, 2, 640,1)
                    cuda_processing.direct_minimize_wrap(markers_over_time_gpu, markers_age_gpu, markers_gpu,2, 640,1, 640)
                    cuda_processing.c_add_wrap(markers_over_time_gpu, ones_gpu,1,640,640)

                    cuda_processing.coharmonic_mean_wrap(markers_over_time_gpu, markers_gpu, 7, 7,2,640,1,640)

                    cuda_processing.scalar_divide_wrap(markers_gpu, 15, 640,1)
                    #cuda_processing.move_to_cpu(markers_gpu, markers_hold)
                    #print(markers_hold)
                    cuda_processing.invert_wrap(markers_gpu,num, 640,1)
                    #cuda_processing.scalar_divide_wrap(markers_gpu, 30, 640,1)
                    cuda_processing.move_to_cpu(markers_gpu, markers_hold)


                    #apply pre warp
                    #cuda_processing.coharmonic_mean_wrap(markers_hold_gpu,markers_gpu,2,2,1,640,1,640)
                    #cuda_processing.direct_conv_wrap(markers_hold_gpu,markers_gpu,lpf_gpu,lpf_np.shape[0],1,640,640)

                    #cuda_processing.direct_conv_wrap(markers_hold_gpu,markers_gpu,lpf_gpu,lpf_np.shape[0],1,640,640)
                    #cuda_processing.scalar_divide_wrap(markers_gpu, 5, 1, 640)

                    cuda_processing.prewarp_wrap(markers_gpu,markers_hold_gpu, 0,310, 640, theta_max, 15,640)
                    #cuda_processing.move_to_cpu(new_vals_ptr,display_np)
                    #cuda_processing.direct_conv_wrap(markers_hold_gpu,markers_gpu,flt_ptr,flt.shape[0],1,640,640)

                    #cuda_processing.age_out_wrap(markers_over_time_gpu, markers_age_gpu, markers_hold_gpu, 2, 20,32,640)
                    #cuda_processing.c_add_wrap(markers_over_time_gpu, markers_hold_gpu,1, 640,640);
                    #cuda_processing.scalar_divide_wrap(markers_over_time_gpu, 2, 640,1)

                    if ROS and True and c ==0:
                        #cuda_processing.move_to_cpu(markers_hold_gpu, markers_hold)

                        laserscan_msg = populate_laserscan(markers_hold[40:-80], dx)
                        # Publish the LaserScan message
                        laserscan_pub.publish(laserscan_msg)
                        c=5

                    #print(markers_hold)
                    #for i,col in enumerate(markers_hold):
                    #    for j in range(1,int(col / 640)+1):
                    #        display_np[int(col - j * 640)] = display_np[int(col)]

                    #print(display_np)
                    #print(display_np[200*400:300*400:400])
                    #conv_arr2_ptr = cuda_processing.move_to_gpu(conv_arr2)
                    #print(markers_hold)
                    #print(display_np.reshape(400,640))
                    #cv2.imshow(
                    #    "raw",
                    #    cv2.hconcat(
                    #        [display_np.reshape(400,640)]
                    #    ),
                    #)
                    cv2.imshow(
                        "scatter", scatterplot((400,640), markers_hold[:, np.newaxis]+0.001)
                    )
                    cv2.waitKey(1)

                    cuda_processing.move_to_gpu_addr(markers_hold_gpu, markers,640)
                    cuda_processing.move_to_gpu_addr(markers_gpu, markers,640)


                c-=1
                continue
                #if FIRST < 0:
                #    narr[-1] = np.minimum(depthFrame, narr[-1])

                if False and not c % RATE and FIRST < 0:  # integrate over 25 frames
                    markers = matrix_processing.process(narr[-1])
                    flipf = np.flip(narr[-1].T, axis=1)
                    
                    for i, elem in enumerate(markers):
                        if int(elem) < CUTOFF:
                            markers[i] = flipf[i, int(elem) + 50]
                            flipf[i, int(elem) + 50 :] = flipf[i, int(elem) + 50]
                        else:
                            markers[i] = num
                    markers = np.flip(markers)
                    # flipf = np.flip(narr[-1].T, axis=1)
                    # holdo = np.arange(len(markers))
                    # less_than_cutoff = markers.astype(int) < CUTOFF
                    # greater_than_cutoff = markers.astype(int) >= CUTOFF
                    # # flipf[ np.arange(len(markers))[less_than_cutoff], markers[less_than_cutoff].astype(int) + 50:] = flipf[less_than_cutoff, markers[less_than_cutoff].astype(int) + 50]
                    # markers[less_than_cutoff] = flipf[np.arange(len(markers))[less_than_cutoff], markers[less_than_cutoff].astype(int) + 50]
                    # # markers[greater_than_cutoff] = num
                    # # flipf[less_than_cutoff, markers[less_than_cutoff].astype(int) + 50:] = flipf[less_than_cutoff, markers[less_than_cutoff].astype(int) + 50]

                    # markers[~less_than_cutoff] = flipf[np.arange(len(markers))[~less_than_cutoff], markers[~less_than_cutoff].astype(int) + 50]

                    # markers = np.flip(markers)
                    prev_markers[0] = np.minimum(markers, prev_markers[0])
                if not c % RATE and FIRST < 0:
                    prev_markers[0] = scalar_outlier_rejection(prev_markers[0])

                    prev_markers[0] = matrix_processing.arr_conv(prev_markers[0]) / 9
                    # prev_markers[0] = scalar_outlier_rejection(prev_markers[0])
                    # prev_markers[0] = scalar_outlier_rejection(nval)

                    x, dx = point_cloud(prev_markers[0])
                    # Populate LaserScan message
                    if ROS and False:
                        laserscan_msg = populate_laserscan(x, dx)
                        # Publish the LaserScan message
                        laserscan_pub.publish(laserscan_msg)

                    if DISPLAY:
                        cv2.imshow(
                            "raw",
                            cv2.hconcat(
                                [depthFrame / 1200, np.flip(flipf, axis=1).T / 1200]
                            ),
                        )

                        cv2.waitKey(1)
                if not c % (RATE):
                    prev_markers = [np.array([10000 for _ in range(640)])]
                if True and not c % RATE and FIRST < 0:
                    narr = [np.zeros((400, 640)) + 10000]
                    c = 0

                FIRST -= 1
                if FIRST < 0:
                    c += 1


a = 0


def getMockIMU(packet):
    global a
    # oops only X
    x = 0  # X IS FRONT AND BACK
    y = 0
    z = 0
    real = 0
    zero = 0
    a += 1  # btw i think its a validation thing similar how it wasnt reading your previous data
    packet.acceleroMeter.x = 1  # linear velocity north/south
    packet.acceleroMeter.y = 0  # linear velocity east/west (when this is the only non zero number it does not look like its moving)
    packet.acceleroMeter.z = 0  # linear velocity up/down
    packet.rotationVector.i = 0  # unk when alone, combined with accel.x turning left/right? but so does j apparently?
    packet.rotationVector.j = 0  # unk
    packet.rotationVector.k = 0  # unk
    packet.rotationVector.real = 0
    packet.gyroscope.x = 0  # unk
    packet.gyroscope.y = 0  # unk
    packet.gyroscope.z = 0  # paralell to floor, circular motion left/right # x=0, y=0, z=10 this makes it do point turns
    return packet


if __name__ == "__main__":
    main()
