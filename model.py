import torch.nn as nn
import torch
import torch.nn.functional as F
"""lenet5_mode"""
class LeNet5(nn.Module):
    def __init__(self,n_classes):
        super(LeNet5,self).__init__()
        """Define the lenet5 architecture """
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1),
            nn.Sigmoid(),
            #1行，6列，5*5，stride=1的卷积核，nn.Tanh双曲正切，output:32*32->28*28*6
            nn.AvgPool2d(kernel_size=2),
            #汇聚层，使用平均汇聚，采样窗口2*2，output：28*28*6->14*14*6
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1),
            nn.Sigmoid(),
            #6行，16列，5*5，stride=1的卷积核，nn.Tanh双曲正切，input:14*14*6，output:10*10*16
            nn.AvgPool2d(kernel_size=2),
            #汇聚层：output:5*5*16
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Sigmoid(),
            # 16行，120列，5*5，stride=1的卷积核，nn.Tanh双曲正切，output:10*10*16->1*1*120
                                               )

        self.classifier=nn.Sequential(
            nn.Linear(in_features=120,out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84,out_features=n_classes)
        )
    def forward(self,x):
        #print(x.shape)
        x=self.feature_extractor(x)
        x=torch.flatten(x,1)
        #print(x.shape)
        logits=self.classifier(x)
        probs=F.softmax(logits,dim=1)
        return logits,probs
if __name__ == '__main__':
    m1=LeNet5(5)
    print(m1)






