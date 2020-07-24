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
def read_frame():
    global raw_frame
    global ret
    global cap
    global frame_read
    global fps
    # Capture frame-by-frame
    timer=time.time()
    print("Start reading")
    while cap is not None:
            ret, raw_frame = cap.read()
            frame_read=True
            #print("frame read")
            if not ret:
                break

            elapse=time.time()-timer  
            #print(elapse)  
            if elapse<1.0/fps:
                time.sleep(1.0/fps-elapse)
            timer=time.time()



# def detect_frame():
#     global raw_frame
#     global processed_frame
#     global current_boxes
#     global detectmode
#     global ret
#     global cap
#     global frame_read
#     global fps
#     global frame_proccessed
#     global model
#     timer=time.time()
#     with torch.no_grad():
#         print("while loop start")
#         while raw_frame is not None:    
            
#             if not ret:  #video ends
#                 break
#             if detectmode and frame_read==True:
#                 frame_read=False
#                 #dataset = LoadImages(source, img_size=img_size)
#                 im0s=raw_frame.copy()
#                 img = letterbox(im0s, new_shape=img_size)[0]
                
#                 # Convert
#                 img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#                 img = np.ascontiguousarray(img)

#                 img_copy=img.copy()
#                 path=source
                
#                 # dataset=[source, img.copy(), img0, cap]
#                 # print (dataset)
#                 t0 = time.time()
#                 print ("To t0: "+str(t0-timer))
#                 img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
#                 _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
#                 #for path, img, im0s, vid_cap in dataset:
#                 img = torch.from_numpy(img_copy).to(device)
#                 img = img.half() if half else img.float()  # uint8 to fp16/32
#                 img /= 255.0  # 0 - 255 to 0.0 - 1.0
#                 if img.ndimension() == 3:
#                     img = img.unsqueeze(0)

#                 t1 = torch_utils.time_synchronized()
#                 print ("To t1: "+str(t1-t0))
#                 pred = model(img, augment=augment)[0]

#                 pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
#                 t2 = torch_utils.time_synchronized()
#                 print ("To t2: "+str(t2-t1))
#                 for i, det in enumerate(pred):  # detections per image
#                     p, s, im0 = path, '', im0s

#                 #save_path = str(Path(out) / Path(p).name)
#                 s += '%gx%g ' % img.shape[2:]  # print string
#                 gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh

#                 if det is not None and len(det):
#                     # Rescale boxes from img_size to im0 size
#                     det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

#                     # Print results
#                     for c in det[:, -1].unique():
#                         n = (det[:, -1] == c).sum()  # detections per class
#                         s += '%g %ss, ' % (n, names[int(c)])  # add to string

#                     # Write results
#                     bboxes=[]
#                     for *xyxy, conf, cls in det:
#                         # if save_txt:  # Write to file
#                         #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         #     with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
#                         #         file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
#                         bbox=[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
#                         bboxes.append(bbox)                   
#                         #if save_img or view_img:  # Add bbox to image
#                         label = '%s %.2f' % (names[int(cls)], conf)
#                         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
#                     #cv2.imshow("sample",im0)
#                     current_boxes=bboxes.copy()
#                     processed_frame=im0
#                 else:
#                     processed_frame=raw_frame.copy()
#                 print("To t3: "+str(time.time()-t2))
#                 cv2.putText(processed_frame, "Not Tracking", (20,20),
#                             cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)

#                 frame_proccessed=True

#                 elapse=time.time()-timer       
#                 if elapse<1.0/fps:
#                     time.sleep(1.0/fps-elapse)
#                 timer=time.time()

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
    timer=time.time()
   
    print("detect")
    if detectmode and frame_read==True:
        frame_read=False
        #dataset = LoadImages(source, img_size=img_size)
        im0s=raw_frame.copy()
        img = letterbox(im0s, new_shape=img_size)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        #img_copy=img.copy()
        path=source
        
        t0 = time.time()
        #print ("To t0: "+str(t0-timer))
                
        img = torch.from_numpy(img).to(device)
        
        img = img.half() if half else img.float()  # uint8 to fp16/32
        
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        

        t1 = torch_utils.time_synchronized()
        #print ("To t1: "+str(t1-t0))
        pred = model(img, augment=augment)[0]
        print ("pred time: "+str(time.time()-t1))
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        
        t2 = torch_utils.time_synchronized()
        print ("To t2: "+str(t2-t1))
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
            processed_frame=raw_frame.copy()
        print("To t3: "+str(time.time()-t2))
        cv2.putText(processed_frame, "Not Tracking", (20,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        frame_proccessed=True

        elapse=time.time()-timer       
        if elapse<1.0/fps:
            time.sleep(1.0/fps-elapse)
        timer=time.time()
       
def track_frame():
    global detectmode
    global tracker
    global raw_frame
    global processed_frame
    global trackmode
    global frame_proccessed
    timer=time.time()
    print("track thread running")
    while raw_frame is not None:
       
        if not ret:  #video ends
            break
        if trackmode==True:
            
            (success,box)=tracker.update(raw_frame)

            if success:
                current_frame=raw_frame.copy()
                (x, y, w, h) = [int(v) for v in box]
                processed_frame=cv2.rectangle(current_frame, (x, y), (x + w, y + h),
                        (0, 255, 0), 2)

                cv2.putText(processed_frame, "CSRT Tracker", (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                frame_proccessed=True

                elapse=time.time()-timer        
                if elapse<1.0/fps:
                    time.sleep(1.0/fps-elapse)
                timer=time.time()
            else:
                trackmode=False
                detectmode=True
        


# def display_frame():
#     global processed_frame
#     global ret
    
#     cv2.imshow('Video',processed_frame)


def checkclick(event,x,y,flags,param):
    global current_boxes
    global detectmode
    global tracker
    global trackmode
    global raw_frame
    for bbox in current_boxes:
        [x1,y1,x2,y2]=bbox
        if x>=x1 and x<=x2 and y>=y1 and y<=y2 and trackmode==False:
            w=x2-x1
            h=y2-y1
            xywh=(x1,y1,w,h)
            tracker.init(raw_frame,xywh)
            detectmode=False
            trackmode=True
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/videos/MOT16-06-raw.webm', help='source')  # file/folder, 0 for webcam
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

        #Initialize cv2 build-in tracker
        tracker=cv2.TrackerCSRT_create()

        #read one frame first to prevent error
        ret, raw_frame = cap.read()

        
        #start threads
        readthread=threading.Thread(target=read_frame)
        readthread.start()

        # with torch.no_grad():
        #     detectthread=threading.Thread(target=detect_frame)
        #     detectthread.start()

        trackthread=threading.Thread(target=track_frame)
        trackthread.start()

        # displaythread=threading.Thread(target=display_frame)
        # displaythread.start()
        counter=0
        #print("Waiting")
        # while frame_proccessed==False:
        #     pass
        # print("Waiting Finished")
        
        while True:   
            if not ret:
                print ("Video finished")
                cap.release()         
                cv2.destroyAllWindows()
                break
            if raw_frame is not None:
                detect_frame()

            if frame_proccessed==True:
                frame_proccessed=False
                cv2.imshow('Video',processed_frame) 
                cv2.waitKey(1)#(int(1000/fps))         
                print ("Frame: "+str(counter))
                counter+=1


        



