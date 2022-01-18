# day 1 정리

## 파이토치 기본
* [파이토치 공식 문서](https://tutorials.pytorch.kr/beginner/basics/intro.html) 
* 주요 내용은 아래와 같음

> 1. 텐서(Tensor)
> 2. Dataset과 DataLoader
> 3. 변형(Transform)
> 4. 신경망 모델 구성하기
> 5. torch.autograd를 사용한 자동 미분
> 6. 모델 매개변수 최적화하기
> 7. 모델 저장하고 불러오기
 
## Training an image classifier

* 순서는 아래와 같음

> 1. Load and normalize the CIFAR10 training and test datasets using torchvision
> 2. Define a Convolution Neural Network & test forward propagation api(torch)
> 3. Define a loss function (MSE, MAE, Binary CrossEntropy, Categorical CrossEntropy..)
> 4. Train the network on the training data
> 5. Test the network on the test data
> 6. [Optimazer](https://ganghee-lee.tistory.com/24)
> 7. ResNet
