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
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F


def test(test_loader,
            model,
        ):
    '''
    測試
    '''
    test_acc = 0
    total_size = 0
    total_correct = 0
    predict = np.array([])
    ans = np.array([])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()    #預測要把model變成eval狀態
    with torch.no_grad():
        for img , label in tqdm.tqdm(test_loader):
            img = img.to(device)
            label = label.to(device)

            out = model(img)
            
            #累加每個batch的loss後續再除step數量
            
            testid_p = out.argmax(dim=1)   
            num_correct = (testid_p==label).sum().item() #該batch在train時預測成功的數量   
            total_correct += num_correct
            total_size += label.size(0)

            #把所有預測和答案計算
            predict = np.append(predict,testid_p.cpu().numpy())
            ans = np.append(ans,label.cpu().numpy())
            
        #print('total_correct:',total_correct)    
        #print('total_size:',total_size)

        test_acc = round((total_correct/total_size)*100,4)
    
    return test_acc,predict.astype(int),ans.astype(int) 

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
		return F.pad(image, padding, 0, 'constant') # type: ignore

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:',device)
    classes_lable = [
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
    CURRENT_PATH = os.path.dirname(__file__)
    BEST_WEIGHT_NAME = f'epoch_14_trainLoss_0.1482_trainAcc_98.44_valLoss_0.4114_valAcc_89.52.pth'
    TMP_ROOT = f'model_weight/model_mobilenet_v3_small'
    SAVE_MODELS_PATH = f'{CURRENT_PATH}/{TMP_ROOT}/{BEST_WEIGHT_NAME}'
    model = model_mobilenet_v3_small(num_classes=16)
    model.load_state_dict(torch.load(SAVE_MODELS_PATH))
    #print(model)
    print('Loading Weight:',BEST_WEIGHT_NAME)
    NEURAL_NETWORK = model.to(device) 

    
    SPLIT_RATIO = 0.2
    SHUFFLE_DATASET = True
    BATCH_SIZE=32

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
        root=f'{CURRENT_PATH}/data/fruit/test',
        transform=test_transform
    )
    # training data loaders
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
    )

    test_acc,predict,ans= test(test_loader=test_loader,
                                model=NEURAL_NETWORK,)
    #print(predict,ans)

    print('test Acc.{}%'.format(test_acc))
    #print(predict,ans)
    print(classification_report(ans,predict))

    plt.rcParams['figure.figsize'] = [20, 20]
    disp = ConfusionMatrixDisplay.from_predictions(
        ans,
        predict,
        display_labels=classes_lable,
        cmap=plt.cm.Blues, # type: ignore
        normalize='true',
    )
    plt.savefig(f'{CURRENT_PATH}/{TMP_ROOT}/ConfusionMatrix.jpg',bbox_inches='tight')
    #plt.show()
    