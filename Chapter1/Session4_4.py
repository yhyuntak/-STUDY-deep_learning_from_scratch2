import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import Trainer
from dataset import spiral
from Session4 import TwoLayerNet


max_epoch = 300
batch_size= 30
input_feature = 2
hidden_size = 4
class_number = 3
learning_rate = 1.0

x,t = spiral.load_data()
model = TwoLayerNet(input_size=input_feature,hidden_size=hidden_size,output_size=class_number)
optimizer = SGD(lr=learning_rate)

trainer = Trainer(model,optimizer)
trainer.fit(x,t,max_epoch,batch_size,eval_interval=10)
trainer.plot()

