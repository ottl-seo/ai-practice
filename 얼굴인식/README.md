# 얼굴인식

제공된 트레이닝 데이터셋을 이용하여 사람의 홍채를 인식하는 인공지능 모델을 구현하고, 구현된 모델을 이용하여 테스트 데이터셋의 홍채를 분류하세요.   
- DT, RF, NN, CNN, LSTM 등 다양한 알고리즘/네트워크 이용 가능
- 존재하는 네트워크(예, ResNet 등)를 변형하여 사용 가능
- Architecture, Layer, Parameter 등 모델의 구성은 자유롭게 설정 가능
- Cross Validation 필수, Fold 수는 자유롭게 설정 가능
- Normalization, Dropout, Ensemble, Auto Encoder 등의 옵션을 자유롭게 사용 가능


### Train set
트레이닝 데이터셋 (https://bit.ly/2PWXD5i)   
• 1명당 3개씩, 350명의 대한 얼굴 데이터, 총 1,050개의 얼굴 데이터   
• Label : 파일명의 첫 4자리 숫자   


### Test set
테스트 데이터셋     
• 총 700개의 얼굴 데이터   
• 하나의 얼굴 데이터가 350명 중 어떤 사람의 얼굴인지를 분류하세요.   
