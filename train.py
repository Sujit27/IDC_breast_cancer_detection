from data import *
from model import *

from torch.utils.data  import SubsetRandomSampler
from torch import optim
from sklearn.metrics import f1_score

import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-lr',type=float,default = 0.001, dest='learning_rate',help='Learning rate')
parser.add_argument('-bs',type=int,default = 32, dest='batch_size',help='Batch size')
parser.add_argument('-ne',type=int,default = 40, dest='num_epochs',help='Number of epochs to train')
parser.add_argument('-dp',type=int,default = 100, dest='display',help='Number of iterations after which to display loss')
parser.add_argument('-wd',type=int,default = 0.0, dest='weight_decay',help='Weight decay factor for regularization')

opt = parser.parse_args()

def main():
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # create dataset
    transform = transforms.Compose([transforms.RandomAffine((-90,90)),transforms.ToTensor()])
    dataset = CancerDataset(transform=transform)
    
    # create network
    input_channel_num = 3
    net = CancerNet(input_channel_num)
    net.to(device)
    
    # split training and validation data and create dataloaders
    validation_split = 0.1
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    shuffle_dataset = True
    random_seed = 0

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    trainloader = DataLoader(dataset, batch_size=opt.batch_size,sampler=train_sampler) 
    validationloader = DataLoader(dataset, batch_size=opt.batch_size,sampler=valid_sampler)
    
    # set loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    
    # Start Training
    print("Training Starts")
    for epoch in range(opt.num_epochs):
        net.train()
        
        for i, data in enumerate(trainloader, 0):
            images,labels = data
            
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if i%opt.display==display-1:
                disp_and_evaluate('train',epoch,i,outputs,labels,loss)               
        
        net.eval()
        for i, data in enumerate(trainloader, 0):
            images,labels = data
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            if i%opt.display==display-1:
                disp_and_evaluate('train',epoch,i,outputs,labels,loss) 
                
    torch.save(net.state_dict(), os.path.join("saved_model","breast_cancer_detector.pth"))


def disp_and_evaluate(phase,epoch,i,outputs,labels,loss):
    if phase=="train":
        print("Epoch: {}, Batch : {}, Training Loss : {}".format(epoch,i,loss.item()))
    elif phase=="validation":
        print("Epoch: {}, Batch : {}, Validation Loss : {}".format(epoch,i,loss.item()))
    labels = labels.detach().cpu().tolist()
    outputs = outputs.detach().cpu().tolist()
    outputs = [1 if output>0.5 else 0 for output in outputs]
    f1_score = f1_score(labels,outputs)
    print("F1 score: {}".format(f1_score))      

if __name__ == "__main__":
    main()