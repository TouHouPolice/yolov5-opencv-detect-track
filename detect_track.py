import argparse
import torch.backends.cudnn as cudnn
from utils import google_utils
from utils.datasets import *
from utils.utils import *
import numpy as np
import cv2
import threading
import time
import torch.multiprocessing as mp
import keyboard

def key_event():
    global trackmode
    global detectmode
    while True:
        if keyboard.is_pressed('q'): 
            print('Shutting down')
            exit()
        if keyboard.is_pressed('c'): 
            print("Quit trackmode")
            trackmode=False
            detectmode=True

def detect_frame():
    global raw_frame
    global processed_frame
    global current_boxes
    global detectmode
    global ret
    global cap
    global frame_read
    global fps
    global frame_proccessed
    global model
    global displaythread
    global trackmode
    global trackername
    global tracker
    timer=time.time()
   
    print("detect")
    dataset = LoadImages(source, img_size=img_size)
    t0 = time.time()
    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap in dataset:
        raw_frame=im0s
        if detectmode :
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = torch_utils.time_synchronized()
            
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s

                #save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    bboxes=[]
                    for *xyxy, conf, cls in det:
                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #     with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                        #         file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                        bbox=[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        bboxes.append(bbox)                   
                        #if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    #cv2.imshow("sample",im0)
                    current_boxes=bboxes.copy()
                    processed_frame=im0
                else:
                    processed_frame=im0
                
                cv2.putText(processed_frame, "Not Tracking", (20,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                frame_proccessed=True
                
        elif trackmode==True:
            #gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            (success,box)=tracker.update(im0s)

            if success:
                print("tracker updated")
                current_frame=im0s
                (x, y, w, h) = [int(v) for v in box]
                processed_frame=cv2.rectangle(current_frame, (x, y), (x + w, y + h),
                        (0, 255, 0), 2)

                cv2.putText(processed_frame, trackername, (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                frame_proccessed=True

                elapse=time.time()-timer        
                if elapse<1.0/fps:
                    time.sleep(1.0/fps-elapse)
                timer=time.time()
            else:
                print("tracker lost")
                tracker=None
                tracker=OPENCV_OBJECT_TRACKERS[trackername]()
                trackmode=False
                detectmode=True
        
        cv2.imshow('Video',processed_frame)
        cv2.waitKey(1)

        elapse=time.time()-timer       
        if elapse<1.0/fps:
            time.sleep(1.0/fps-elapse)
        timer=time.time()
       

def checkclick(event,x,y,flags,param):
    global current_boxes
    global detectmode
    global tracker
    global trackmode
    global raw_frame
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print ("x: "+str(x))
        print ("y: "+str(y))
        print ("")
        for bbox in current_boxes:
            [x1,y1,x2,y2]=bbox
            if x>=x1 and x<=x2 and y>=y1 and y<=y2 and trackmode==False:
                w=x2-x1
                h=y2-y1
                xywh=(x1,y1,w,h)
                #gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
                tracker.init(raw_frame,xywh)
                detectmode=False
                trackmode=True
                print("bbox: "+str(bbox))
                print("xywh: "+str(xywh))
                print("tracker initialized")
                break
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/videos/MOT16-06.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    #variables for convenience
    weights=opt.weights
    source=opt.source             
    output=opt.output
    img_size=opt.img_size
    conf_thres=opt.conf_thres
    iou_thres=opt.iou_thres
    fourcc=opt.fourcc
    device=opt.device
    view_img=opt.view_img
    save_txt=opt.save_txt
    agnostic_nms=opt.agnostic_nms
    classes=opt.classes
    augment=opt.augment#opt.augment
    save_img=False

    with torch.no_grad():
        #Setup model
        device = torch_utils.select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
        model.to(device).eval()
        print ("half: "+str(half))
        if half:
            model.half()  # to FP16

        #for visualization
        names = model.names if hasattr(model, 'names') else model.modules.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
        #Initialize global variables
        raw_frame=None
        processed_frame=None
        detectmode=True
        trackmode=False
        ret=0
        current_boxes=[]
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)

        #variables to keep consistent frame rate
        frame_read=False
        frame_proccessed=False

        #Initialize cv2 window
        
        cv2.namedWindow("Video")
        cv2.setMouseCallback("Video",checkclick)


        OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	    }

        #Initialize cv2 build-in tracker
        trackername="medianflow"
        tracker=OPENCV_OBJECT_TRACKERS[trackername]()


        

        counter=0
        detect_frame()
        
        

        



