import torch
import argparse
import gradio as gr
from PIL import Image
from numpy import random
from pathlib import Path
import os
import time
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
import cv2
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier,scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


os.system('git clone https://github.com/WongKinYiu/yolov7')


 
def Custom_detect(img):
    model='best' 
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=model+".pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='Temp_file/', help='source') 
    parser.add_argument('--img-size', type=int, default=100, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    opt = parser.parse_args()
    img.save("Temp_file/test.jpg")
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = True 
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    #print("webcam",webcam)
    #webcam = True
    save_dir = Path(increment_path(Path(opt.project)/opt.name,exist_ok=opt.exist_ok))
    
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu' 
    model = attempt_load(weights, map_location=device) 
    stride = int(model.stride.max())  
    imgsz = check_img_size(imgsz, s=stride)
    if trace:
        model = TracedModel(model, device, opt.img_size)
    if half:
        model.half() 
        
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True 
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) 
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() 
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        pred = non_max_suppression(pred,opt.conf_thres,opt.iou_thres,classes=opt.classes, agnostic=opt.agnostic_nms) 
        t2 = time_synchronized()
        
        
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
    
        for i, det in enumerate(pred): 
            if webcam: 
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  

                
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh) 
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  

            if save_img:
                if dataset.mode == 'image':                
                    cv2.imwrite(save_path, im0)
                else: 
                    if vid_path != save_path: 
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap: 
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else: 
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''

    print(f'Done. ({time.time() - t0:.3f}s)')

    return Image.fromarray(im0[:,:,::-1])
inp = gr.Image(type="pil")
output = gr.Image(type="pil")

examples=[["Examples/Image1.jpg","Image1"],["Examples/Image2.jpg","Image2"]]

io=gr.Interface(fn=Custom_detect, inputs=inp, outputs=output, title='Detección de baches',examples=examples,cache_examples=False)
io.launch()

