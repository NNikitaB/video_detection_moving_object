import numpy as np
import cv2


#https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
#https://www.geeksforgeeks.org/getting-started-with-object-tracking-using-opencv/
#https://docs.opencv.org/5.x/d4/dee/tutorial_optical_flow.html
#https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html

#Алгоритмы захвата и отслеживания изображения
#*   Motion Templates
#*   Mean-Shif подходит
#*   CamShif
#*   Lucas – Kanade ( Lucas–Kanade optical flow и KLT Kanade–Lucas–Tomasi))подходит
#*   Viola – Jone
#* optical flow
#* Детектирование ключевых точек (feature detection ) такие как SIFT (Scale-Invariant Feature Transform), SURF (Speeded Up Robust Features) и ORB (Oriented FAST and Rotated BRIEF).
#*  YOLO (You Only Look Once) и SSD (Single Shot Detector)


def video_exec_save(video_path,f,save_path):
  normalize = 0
  res = f(video_path)


path = './video/'
execution = '.mp4'
dct_video = {'flow': 'flow_10s'}



vp = path + dct_video['flow'] + execution

def detect_optic_flow(video_path):


    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2',rgb)
        #cv2_imshow(rgb)


        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()
