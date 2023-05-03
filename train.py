import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
import os,math,shutil,tqdm
import numpy as np
from network import model_resnet50,model_resnet101,model_densenet121,model_densenet161,model_densenet201,\
model_efficientnet_b0, model_efficientnet_b4,model_efficientnet_b7,model_efficientnet_v2_s,model_efficientnet_v2_l,\
model_mobilenet_v3_small
from early_stop import early_stop
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from torchsummary import summary

def train(train_loader,
          valid_loader,
          model,
          batch_size,
          optimizer,
          loss_func,
          epoch,
          ):
    '''
    訓練
    '''
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    best_acc = 0
    best_loss = 0
    best_val_acc = 0
    best_val_loss = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for num_epoch in range(epoch):
        train_avg_loss = 0                #每個epoch的loss
        train_avg_acc = 0                 #每個epoch的acc
        total_acc = 0
        for step, (img, label) in tqdm.tqdm(enumerate(train_loader)):
            #確保每一batch都能進入model.train模式
            model.train()
            #放置gpu訓練
            img = img.to(device)
            label = label.to(device)
            #img經過nural network卷積後的預測(前向傳播),跟答案計算loss 
            out = model(img)
            loss = loss_func(out,label)
            #優化器的gradient每次更新要記得初始化,否則會一直累積
            optimizer.zero_grad()
            #反向傳播偏微分,更新參數值
            loss.backward()
            #更新優化器
            optimizer.step()

            #累加每個batch的loss後續再除step數量
            train_avg_loss += loss.item()
            
            #計算acc
            train_p = out.argmax(dim=1)                 #取得預測的最大值
            num_correct = (train_p==label).sum().item() #該batch在train時預測成功的數量
            batch_acc  = num_correct / label.size(0)
            total_acc += batch_acc


        val_avg_loss,val_avg_acc = valid(
            valid_loader=valid_loader,
            model=model,
            loss_func=loss_func
        )    
        
        train_avg_loss = round(train_avg_loss/len(train_loader),4)   #該epoch每個batch累加的loss平均
        train_avg_acc = round(total_acc/len(train_loader),4)         #該epoch的acc平均

        train_loss.append(train_avg_loss)
        train_acc.append(train_avg_acc)
        valid_loss.append(val_avg_loss)
        valid_acc.append(val_avg_acc)

        print('Epoch: {} | train_loss: {} | train_acc: {}% | val_loss: {} | val_acc: {}%'\
              .format(num_epoch, train_avg_loss,round(train_avg_acc*100,4),val_avg_loss,round(val_avg_acc*100,4)))
        
        #early stop
        performance_value = [num_epoch,
                             train_avg_loss,
                             round(train_avg_acc*100,4),
                             val_avg_loss,
                             round(val_avg_acc*100,4)]
        EARLY_STOP(val_avg_acc,
                   model=model,
                   performance_value = performance_value
                   )
        
        if EARLY_STOP.early_stop:
            print('Earlt stopping')
            break    


    return train_loss,train_acc,valid_loss,valid_acc 

def valid(valid_loader,
            model,
            loss_func,
            ):
    '''
    訓練時的驗證
    '''
    val_avg_loss = 0
    val_avg_acc = 0
    total_acc = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()    #預測要把model變成eval狀態
    with torch.no_grad():
        for img , label in valid_loader:
            img = img.to(device)
            label = label.to(device)

            out = model(img)
            loss = loss_func(out,label)
            #累加每個batch的loss後續再除step數量
            val_avg_loss += loss.item()

            valid_p = out.argmax(dim=1)   
            num_correct = (valid_p==label).sum().item() #該batch在train時預測成功的數量   
            batch_acc  = num_correct / label.size(0)
            total_acc += batch_acc
            

        val_avg_loss = round(val_avg_loss/len(valid_loader),4)
        val_avg_acc = round(total_acc/len(valid_loader),4)

    return val_avg_loss,val_avg_acc

def plot_statistics(train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                    SAVE_MODELS_PATH):
    '''
    統計train、valid的loss、acc
    '''
    fig, ax = plt.subplots()
    epcoh = [x for x in range(len(train_loss))]
    ax2 = ax.twinx()
    t_loss = ax.plot(train_loss,color='green',label='train_loss')
    v_loss = ax.plot(valid_loss,color='red',label='valid_loss')
    t_acc = ax2.plot(train_acc,color='#00FF55',label='train_acc')
    v_acc = ax2.plot(valid_acc,color='#FF5500',label='valid_acc')
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2.set_ylabel("acc")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig(f'{SAVE_MODELS_PATH}/train_statistics',bbox_inches='tight')
    plt.figure()
    
class SquarePad:
	'''
	方形填充到長寬一樣大小再來resize
    '''
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = [hp, vp, hp, vp]
		return F.pad(image, padding, 0, 'constant')


if __name__ == '__main__':
    '''
    修改model train:
    model都有import在上面
    SAVE_MODELS_PATH 將後面的檔案名稱換成要load model的名稱 ex:model_resnet101、model_densenet121

    '''
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:',device)
    
    CURRENT_PATH = os.path.dirname(__file__)
    NEURAL_NETWORK = model_resnet101(num_classes=16).to(device)      #讀不同的model
    #print(summary(NEURAL_NETWORK, input_size=(3,224,224)))
    SHUFFLE_DATASET = True
    BATCH_SIZE=32
    SAVE_MODELS_PATH = f'{CURRENT_PATH}/model_weight/model_resnet101' #記得改這行的model_resnet101
    
    try:
        shutil.rmtree(SAVE_MODELS_PATH)
    except:
        pass
    os.makedirs(SAVE_MODELS_PATH)

    LEARNING_RATE = 0.001 #lambda x: ((1.001 + math.cos(x * math.pi / EPOCH))) #* (1 - 0.1) + 0.1  # cosine
    EPOCH = 1000
    
    OPTIMIZER = torch.optim.Adam(NEURAL_NETWORK.parameters(), lr=LEARNING_RATE)
    LOSS = nn.CrossEntropyLoss()


    EARLY_STOP = early_stop(save_path=SAVE_MODELS_PATH,
                            mode='max',
                            monitor='val_acc',
                            patience=5)
    
    train_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.ColorJitter(brightness=0.2,contrast=0.2, saturation=0.2, hue=0.2), # type: ignore
        #transforms.RandomRotation(30),
        transforms.ToTensor(),  
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # the validation transforms
    valid_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # training dataset
    train_dataset = datasets.ImageFolder(
        root=f'{CURRENT_PATH}/data/fruit/train',
        transform=train_transform
    )
    # validation dataset
    valid_dataset = datasets.ImageFolder(
        root=f'{CURRENT_PATH}/data/fruit/valid',
        transform=valid_transform
    )
    # training data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
    )
    # validation data loaders
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
    )
    
    train_loss,train_acc,valid_loss,valid_acc = train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=NEURAL_NETWORK,
        batch_size=BATCH_SIZE,
        optimizer=OPTIMIZER,
        loss_func=LOSS,
        epoch=EPOCH
    )

    plot_statistics(train_loss,train_acc,valid_loss,valid_acc,SAVE_MODELS_PATH)