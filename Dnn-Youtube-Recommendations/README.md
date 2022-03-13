# Deep Neural Networks for YouTube Recommendations

## Tools
- Langauge: `python3`
- Framework: `pytorch`

## Developer Engineer
- Main: [Gyeongtae Im (LinkedIn Profile)](https://www.linkedin.com/in/gangtaro/)
- Contributor: `None`

## References
- Papers: [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf)
- Dataset: [MovieLens](https://grouplens.org/datasets/movielens/)

---
## How to Build?
### 1. **Candidate Generation** part
수백만개의 영화(item) 중, 수백개의 영화로 후보군을 추려서 전달해줄 부분.  
(주어진 유저에 대해서 해당 유저가 좋아할만한 몇 백개의 영화(item)을 선정해준다.)  

#### **training part**
특정시간(t)에 어떤유저(U)가 context(C)를 가지고 있을 때, 각각의 Video를 볼 확률을 계산하는 문제이다. 즉, 수백만개의 영화(item)의 **Softmax** 값을 계산해야하는 **Extreme-Classification** task이다.  
> **(issue)** Dataset을 구성할 때, Label은 뭘로 해야하고 Loss는 어떻게 계산해줘야하는가?  
DNN 끝 단에서, 결과로 나올 **User vector(1차 output)** 으로 extreme-classification layer (linear + softmax)에 선호할만한 영화(item)를 예측하는데, 이 값이 Softmax로 나오기 때문에, 결과적으로 y label은 해당 유저가 시청했던 영화의 one-hot vector로 구성하는 것이 맞다고 생각함


1. **Watch vector, Search vector**  
유저(U)가 시청했던 영화(items)들의 embeding vectors의 평균, 검색했던 검색어(tokens)의 embeding vectors의 평균을 각각  watch vector, search vector라고 한다.  
    - 그런데, 어떻게 embeding vector를 만들것인가?  
        - embedded video watches:
            - (기본) 그냥 단어 자체로 `nn.Embedding` 사용한다.
            - (TODO) 영화의 다른 정보들도 이용하여 개별 Embeding vector를 구성한다. 
        - embedded seach tockens:
            - (기본) 그냥 단어 자체로 `nn.Embedding` 사용한다.
            - (TODO) Word2Vec을 사용하여, 임베딩을 만들어낸다. 

2. **Demographic & Geographic**  
적절한 방법으로 Embeding을 실시해준다.
    - Demographic: 'age', 'sex', 'occupation'
    - Geographic: 'zip_code'

3. **Example age**  
**FRESHNESS!!!** 과거 데이터에 너무 편중되지 않게, 현재 시점에 가까운 데이터에 가까운 데이터를 더 많이 학습 시키기 위해서 만들어주는 Feature
    >  **(issue)** 어떤 방법으로 최근 데이터를 더 많이 학습하지?  
    DataLoader를 활용하여 현재 시점에 가까운 데이터를 조금 더 많이 나오도록 조절한다(?)


4. Concatenate  
만들어준 Vector를 모두 Concatenate 해서, 하나의 Feature vector로 만들고, 그것을 DNN 아키텍처를 통해서 학습 시킨다. 이때 DNN에서는 **User vector(1차 output)** 가 output으로 설정해준다. 

5. Training




    



---
## Develop-Log
- (2022.03.13): Reading Paper, and Drawing base architecture, Making BaseDataset class in Dataset.py
- (2022.03.13): How to get embed aggregate !