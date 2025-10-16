import numpy as np
import random

def crop_face(img,landmark=None,bbox=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
    assert phase in ['train','val','test']

    #crop face------------------------------------------
    H,W=len(img),len(img[0])

    assert landmark is not None or bbox is not None

    H,W=len(img),len(img[0])
    
    x0_list = []
    x1_list = []
    y0_list = []
    y1_list = []
    
    for i in range(bbox.shape[0]):

        x0_list.append(bbox[i][0][0])
        y0_list.append(bbox[i][0][1])
        x1_list.append(bbox[i][1][0])
        y1_list.append(bbox[i][1][1])
    
    x0, y0, x1, y1 = np.min(x0_list), np.min(y0_list), np.max(x1_list), np.max(y1_list)
    
    if crop_by_bbox:
        w=x1-x0
        h=y1-y0
        
        w0_margin=w/12 #0#np.random.rand()*(w/8)
        w1_margin=w/12
        h0_margin=h/12 #0#np.random.rand()*(h/5)
        h1_margin=h/12
        
        # w0_margin=0 #0#np.random.rand()*(w/8)
        # w1_margin=0
        # h0_margin=0 #0#np.random.rand()*(h/5)
        # h1_margin=0
        
    else:
        x0,y0=landmark[:68,0].min(),landmark[:68,1].min()
        x1,y1=landmark[:68,0].max(),landmark[:68,1].max()
        w=x1-x0
        h=y1-y0
        w0_margin=w/8 #0#np.random.rand()*(w/8)
        w1_margin=w/8
        h0_margin=h/2 #0#np.random.rand()*(h/5)
        h1_margin=h/5

    # if margin: # 进行换脸之前原始坐标情况下的放大
    #     w0_margin*=4
    #     w1_margin*=4
    #     h0_margin*=2
    #     h1_margin*=2
        
    # elif phase=='train': #进行换脸之后
    #     w0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
    #     w1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
    #     h0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
    #     h1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()	
    # else:
    #     w0_margin*=0.5
    #     w1_margin*=0.5
    #     h0_margin*=0.5
    #     h1_margin*=0.5
    
    # if phase=='train': #进行换脸之后
    #     w0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
    #     w1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
    #     h0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
    #     h1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()	
    # else:
    #     w0_margin=w/12
    #     w1_margin=w/12
    #     h0_margin=h/12
    #     h1_margin=h/12
    
     # 根据得到的人脸坐标按照margin进行放大
     		
    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin) + 1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)
    
    img_cropped=img[y0_new:y1_new, x0_new:x1_new]
 
    if landmark is not None:
        landmark_cropped = np.zeros_like(landmark)
        for i,(p,q) in enumerate(landmark):
            landmark_cropped[i]=[p-x0_new,q-y0_new]
    else:
        landmark_cropped=None

    bbox_cropped=None

    if only_img:
        return img_cropped
    
    if abs_coord:
        return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
    else:
        return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)


def generate_random_index(list, message_length):
    idx = random.sample(list, message_length)
    idx.sort()
    idx = random.sample(idx, message_length)
    
    return idx