import numpy as np 

box1 = [-1, 0, 10, 10]
box2 = [2, 2, 1, 1]

def compute_iou(box1,box2):
    in_w=min(box1[2],box2[2])-max(box1[0],box2[0])
    in_h=min(box1[3],box2[3])-max(box1[1],box2[1])

    if(in_w<0 or in_h<0):
        inner = 0
    else:
        inner =in_w*in_h

    outer = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inner

    iou= inner / outer
    print(iou)
    return iou

def xywh_xyxy(box1):
    xyxy=[0,0,0,0]
    xyxy[0]=box1[0]
    xyxy[1]=box1[1]
    xyxy[2]=box1[0]+box1[2]
    xyxy[3]=box1[1]+box1[3]
    return xyxy

def xyxy_xywh(box1):
    xywh = [0,0,0,0]
    xywh[0] = box1[0]
    xywh[1]=box1[1]
    xywh[2]= box1[2]-box1[0]
    xywh[3]= box1[3]-box1[1]
    return xywh

print(compute_iou(box1,box2))
print(xywh_xyxy(box1))
print(xyxy_xywh(box2))