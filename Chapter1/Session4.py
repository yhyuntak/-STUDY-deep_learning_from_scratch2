
import sys
sys.path.append("..")
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from Session4_model import TwoLayerNet

x,t = spiral.load_data()

plt.figure(figsize=(7,5))

zero_ = t[:,0] == 1
one_ = t[:,1] == 1
two_ = t[:,2] == 1

x_1 = x[zero_,:]
x_2 = x[one_,:]
x_3 = x[two_,:]

# plt.scatter(x=x_1[:,0],y=x_1[:,1],marker='x',c='b')
# plt.scatter(x=x_2[:,0],y=x_2[:,1],marker='v',c='g')
# plt.scatter(x=x_3[:,0],y=x_3[:,1],marker='o',c='r')
#
# plt.title("Data plotting")
# plt.show()

"""
학습 코드
"""
max_epoch = 300
batch_size= 30
input_feature = 2
hidden_size = 4
class_number = 3
learning_rate = 1.0

model = TwoLayerNet(input_size=input_feature,hidden_size=hidden_size,output_size=class_number)
optimizer = SGD(lr=learning_rate)

data_size = len(x)
max_iters = data_size//batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):

    # 데이터 랜덤하게 뒤섞기
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]


    for iters in range(max_iters):
        # iter를 돌면서 배치사이즈만큼씩 학습하기.
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]
        """
        forward가 진행되면서 loss가 최종 출력되지만, 각 layer객체마다 파라미터 W,b와 쓰이는 데이터들을 저장해둔다. 
        이때는 아직 gradient가 업데이트가 되지 않은 상태이다.
        """
        loss = model.forward(batch_x,batch_t)
        """
        backward를 진행하면서 각 layer 객체들마다의 backward 메소드를 통해 역전파가 진행되고 객체들의 gradient들에 값이 저장된다.
        이때도 아직 내가 만든 TwoLayerNet의 gradient가 업데이트가 되지 않은 상태이다. 
        그니까 다시 말하기위해 현재 첫번째 iteration이라고 가정하면, 가중치,편향이 초기화된 상태일 뿐이다. ->model.params 
        model.backward()를 통해서 각각의 layer들만! gradient가 계산이 된거다. 이것이 model.grads를 통해서 전부 불러들일 수 있는 것이다. 
        """
        model.backward()
        """
        아래의 optimizer로 지정된 알고리즘을(지금은 SGD)를 이용해서 내 모델의 params를 업데이트한다. 이제야 이해간다.
        """
        optimizer.update(model.params,model.grads)

        # 여기부턴 뭐 loss를 저장했다가 어떤 iteration에 도달하면 평균 loss를 계산해서 저장하고 초기화하는 그런거다.
        total_loss += loss
        loss_count += 1
        if (iters+1) % 10 == 0:
            avg_loss = total_loss /loss_count
            print(": 에폭 %d : 반복 %d / %d : 손실 %.2f" %(epoch+1,iters+1,max_iters,avg_loss))
            loss_list.append(avg_loss)
            total_loss,loss_count = 0,0

"""
이건 그냥 책의 예제를 그대로 따옴. 
"""

# 학습 결과 플롯
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('반복 (x10)')
plt.ylabel('손실')
plt.show()

# 경계 영역 플롯
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# 데이터점 플롯
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()


