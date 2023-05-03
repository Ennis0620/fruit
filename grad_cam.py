import torch,torchvision
import torch.nn as nn
import numpy as np
import cv2,os,tqdm,math
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import shutil
from network import model_resnet50,model_resnet101,model_densenet121,model_densenet161,model_densenet201,\
model_efficientnet_b0, model_efficientnet_b4,model_efficientnet_b7,model_efficientnet_v2_s,model_efficientnet_v2_l,\
model_mobilenet_v3_small

class Grad_cam(nn.Module):
    def __init__(self,model,num_classes):
        super(Grad_cam,self).__init__()
        self.model = model
        self.layers = list(model.children())
        self.grad_block = []
        self.fmap_block = []
        self.num_classes = num_classes

    def backward_hook(self,
                      module,
                      grad_in,
                      grad_out):
        self.grad_block.append(grad_out[0].detach())
    
    def forward_hook(self,
                     module,
                     input,
                     output):
        self.fmap_block.append(output)
    
    def show_cam_on_img(self,
                        img,
                        mask,
                        save_path):
        '''
        將grad_cam的注意力mask畫在原本的img上
        '''
        #H,W,_ = img.shape
        #resize_h,resize_w = 128,128
        
        img = ((img + 1)/2)*255 #unnormalize to 0~255
        
        img = np.uint8(img.numpy())
        img = img.transpose(1,2,0) # type: ignore
    
        heatmap = cv2.applyColorMap(np.uint8(255*mask),cv2.COLORMAP_JET) # type: ignore
        cam_img= np.uint8(0.3*heatmap + 0.7*img)

        img = cv2.resize(img,(64,64))
        heatmap = cv2.resize(heatmap,(64,64))
        cam_img = cv2.resize(cam_img,(64,64)) # type: ignore


        space = np.ones((cam_img.shape[0],5,cam_img.shape[2]))
        con = np.concatenate([img,space,heatmap,space,cam_img],axis=1)
        
        #用plt show圖片
        plt.figure()
        con = cv2.cvtColor(np.float32(con/255),cv2.COLOR_BGR2RGB)# type: ignore #BGR->RGB pixel range:0~1
        im_ratio = con.shape[0]/con.shape[1]
        plt.imshow(con,cmap='jet')
        plt.colorbar(fraction=0.047*im_ratio)
        plt.savefig(save_path,bbox_inches='tight')

    def show_cam_on_img_numpy(self,
                        con,
                        save_path,
                        prob):
        '''
        將grad_cam的注意力mask畫在原本的img上
        '''
        
        #用plt show圖片
        plt.figure()
        con = cv2.cvtColor(np.float32(con/255),cv2.COLOR_BGR2RGB)# type: ignore #BGR->RGB pixel range:0~1
        im_ratio = con.shape[0]/con.shape[1]
        plt.imshow(con,cmap='jet')
        plt.colorbar(fraction=0.047*im_ratio)
        plt.xlabel(str(prob))
        plt.savefig(save_path,bbox_inches='tight')

    def img_preprocess(self,img):
        '''
        '''
        img_in = img.copy()     # type: ignore
        img_in = img[:,:,::-1] # type: ignore
        img_in = np.ascontiguousarray(img_in)
        transform = torchvision.transforms.Compose(
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)) # type: ignore
        )
        img_out = transform(img_in)
        img_out = img_out.squeeze(0)
        return img_out

    def imshow(self,
               img,
               save_path):
        '''
        顯示test_loader的影像
        '''
        img = img / 2 + 0.5 #unnormalizate
        npimg = img.numpy()
        npimg = np.transpose(npimg,(1,2,0))
        plt.imshow(npimg)
        plt.savefig(f'{save_path}')
        #plt.show()

    def comp_class_vec(self,output_vec, index=None):
        """
        计算类向量
        :param ouput_vec: tensor
        :param index: int，指定类别
        :return: tensor
        """
        if not index:
            index = np.argmax(output_vec.cpu().data.numpy())
        else:
            index = np.array(index)
        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = torch.zeros(1, self.num_classes).scatter_(1, index, 1)
        one_hot.requires_grad = True
        class_vec = torch.sum(one_hot.to(device) * output_vec)  # one_hot = 11.8605
        return class_vec       
    
    def generate_grad_cam_focus(self,
                                fmap,
                                grad,
                                shape):
        '''
        依gradient和featuremap生成cam \n
        `fmap`:np array [C,H,W] \n
        `grad`:np array [C,H,W] \n
        @return \n
        np.array [H,W]
        '''
        
        cam = np.zeros(fmap.shape[1:],
                        dtype = np.float32)
        #print('cam zero:',cam.shape)
        weights = np.mean(grad,axis=(1,2))
        #print('weights:',weights.shape)

        for i,w in enumerate(weights):
            #print('w:',w)
            cam += w * fmap[i, :, :]
 
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, shape)
        cam -= np.min(cam)
        cam /= np.max(cam)
        
        return cam
class SquarePad:
	'''
	方形填充到長寬一樣大小再來resize
    '''
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')     # type: ignore

if __name__ == "__main__":
    

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    CURRENT_PATH = os.path.dirname(__file__)
    SAVE_GRAD_CAM_VISUAL = f'{CURRENT_PATH}/grad_cam_visual'
    try:
        shutil.rmtree(SAVE_GRAD_CAM_VISUAL)
    except:
        pass
    os.makedirs(SAVE_GRAD_CAM_VISUAL)
    SPLIT_RATIO = 0.2
    SHUFFLE_DATASET = False
    BATCH_SIZE=1000

    LOSS = nn.CrossEntropyLoss()
   

    BEST_WEIGHT_NAME = f'epoch_8_trainLoss_0.2483_trainAcc_96.15_valLoss_0.2337_valAcc_100.0.pth'
    TMP_ROOT = f'model_weight/model_efficientnet_v2_l_no_none'
    SAVE_MODELS_PATH = f'{CURRENT_PATH}/{TMP_ROOT}/{BEST_WEIGHT_NAME}'
    num_classes = 15
    
    #resnet 最後一層叫fc   densenet叫classifier
    '''
    model = torchvision.models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(model.layer4[2].conv3)
    '''
    model = torchvision.models.efficientnet_v2_l()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs,  num_classes) # type: ignore
    #model.fc = nn.Linear(num_ftrs, 16)
    #print(model.features[-1][0])
    
    
    model.load_state_dict(torch.load(SAVE_MODELS_PATH))
    
    #print(model.features[-1][0])

    net = model.to(device)
    test_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # test dataset
    test_dataset = datasets.ImageFolder(
        root=f'{CURRENT_PATH}/data/fruit/test_1',
        transform=test_transform
    )
    # training data loaders
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
    )

    classes = [
        'apples',
        'bananas',
        'cantaloupes',
        'grapes',
        'guava',
        'kiwifruit',
        'limes',
        'mangos',
        'oranges',
        'pineapples',
        'watermelons',
        'papayas',
        'dragonfruit',
        'durian',
        'sugarapple',
        'none',
    ]

    test_iter = iter(test_loader)
    img,label = next(test_iter)
    show_img = torchvision.utils.make_grid(img,
                                           nrow=int(math.sqrt(BATCH_SIZE)))
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                ])
    origin_imgs = invTrans(img)
   
    for idx,(per_img,per_label) in tqdm.tqdm(enumerate(zip(img,label))):
        
        grad_cam = Grad_cam(model=net,
                            num_classes =  num_classes)    
        
        #grad-cam需要最後一捲積層前向&反向傳播
        #efficientnet_v2_l
        grad_cam.model.features[-1][0].register_forward_hook(grad_cam.forward_hook) # type: ignore
        grad_cam.model.features[-1][0].register_backward_hook(grad_cam.backward_hook) # type: ignore

        #grad_cam.model.layer4[2].conv3.register_forward_hook(grad_cam.forward_hook) # type: ignore
        #grad_cam.model.layer4[2].conv3.register_backward_hook(grad_cam.backward_hook) # type: ignore
        #print(grad_cam.model.conv2)
    
        grad_cam.imshow(show_img,
                    save_path=f'{SAVE_GRAD_CAM_VISUAL}/origin.jpg')
    
        feed2network_img = np.expand_dims(per_img,axis=0)#給network必須為4維(張數,C,H,W)
        feed2network_img = torch.from_numpy(feed2network_img).to(device)
        feed2network_label = np.expand_dims(per_label,axis=0)#給network必須為4維(張數,C,H,W)
        feed2network_label = torch.from_numpy(feed2network_label).to(device)
        
        #forward
        net.eval()
        out = net(feed2network_img).to(device)
        #print("predict: {}".format(classes[idx]))
        #backward
        net.zero_grad()
        #loss = LOSS(out,feed2network_label)
        loss = grad_cam.comp_class_vec(out)
        loss.backward()

        
        #test_pred = out.argmax(dim=1)                 # type: ignore #取得預測的最大值
        out = torch.softmax(out,dim=1)
        test_pred = out.argmax(dim=1)
        prob = torch.max(out).detach().cpu().numpy()
        

        
        grad = grad_cam.grad_block[0].cpu().data.numpy().squeeze()
        fmap = grad_cam.fmap_block[0].cpu().data.numpy().squeeze()
        cam = grad_cam.generate_grad_cam_focus(fmap=fmap,
                                                grad=grad,
                                                shape = (224,224))
        #print(fmap.shape)
        #print(grad.shape)
        #print(cam.shape)
        predict = classes[test_pred.cpu().numpy()[0]]
        real_ans = classes[per_label]
        if prob < 0.5:
            predict = "none"

        save_path = f'{SAVE_GRAD_CAM_VISUAL}/pred_{predict}_real_{real_ans}_{idx}.jpg'
        

        origin_img = (origin_imgs[idx].numpy().transpose(1,2,0))
        origin_img = np.uint8(origin_img*255)
        origin_img = cv2.cvtColor(origin_img,cv2.COLOR_RGB2BGR) # type: ignore
        heatmap = cv2.applyColorMap(np.uint8(cam*255),cv2.COLORMAP_JET) # type: ignore
        cam_img= 0.5*heatmap + 0.5*origin_img # type: ignore
        cam_img = np.uint8((cam_img/cam_img.max())*255)
        space = np.ones((cam_img.shape[0],5,cam_img.shape[2]),dtype = np.uint8) # type: ignore
        con = np.concatenate([origin_img,space,heatmap,space,cam_img],axis=1)
        #cv2.imshow('',con)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        prob = round(prob*100,3)
        

        prob_str = f"prob:{prob}% \n pred:{predict}, real:{real_ans}"
        grad_cam.show_cam_on_img_numpy(con=con,
                                       save_path=save_path,
                                       prob=prob_str)

        
        #img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB)
        
        #grad_cam.show_cam_on_img(img=origin_img,
        #                        mask=cam,
        #                        save_path=save_path)


    
