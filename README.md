## Projects

### 1. Spark PageRank Analysis

**File:** `src/spark_pagerank_analysis.py`

#### 개요
PySpark를 이용해 그래프 데이터를 읽고, dead end와 spider trap을 탐지한 뒤, PageRank를 계산하여 상위 중요 노드를 출력하는 프로젝트입니다.

#### 구현 내용
- 그래프 edge 데이터 파싱 및 중복 제거
- 전체 노드 집합과 out-degree 계산
- dead end 탐지
- self-loop 및 mutual-loop 기반 spider trap 탐지
- PageRank iterative update 구현
- 수렴 조건을 만족할 때까지 rank 갱신
- 상위 PageRank 노드 출력

#### 배운 점
- PySpark를 이용해 그래프 알고리즘을 분산 환경에서 구현하는 방법
- dead end와 spider trap이 PageRank에 어떤 영향을 주는지 이해
- teleportation과 leakage 보정이 필요한 이유 학습
- 반복적 랭킹 알고리즘의 수렴 조건을 코드로 다루는 방법 경험

---

### 2. Neural Network MNIST Classifier

**File:** `src/neural_network_mnist_classifier.py`

#### 개요
CSV 형태의 이미지 데이터를 입력으로 받아, 완전연결 신경망을 직접 구현하고 미니배치 학습을 통해 분류 성능을 평가하는 프로젝트입니다.

#### 구현 내용
- CSV 데이터 로드 및 라벨 분리
- one-hot encoding 적용
- 입력층-은닉층-출력층 구조의 fully connected neural network 구현
- sigmoid activation 함수 적용
- forward propagation 및 backward propagation 구현
- 미니배치 기반 학습 수행
- train accuracy와 test accuracy 출력

#### 배운 점
- 신경망의 forward / backward propagation 흐름을 직접 구현하며 이해
- 가중치 초기화와 learning rate가 학습에 미치는 영향 학습
- one-hot encoding과 분류 문제의 출력 표현 방식 이해
- 딥러닝 프레임워크 없이도 기본적인 분류 모델을 구현할 수 있는 경험 축적

---

### 3. Node2Vec Skip-Gram Embedding

**File:** `src/node2vec_skipgram_embedding.py`

#### 개요
그래프에서 node2vec 방식의 random walk를 생성하고, 이를 skip-gram 방식으로 학습하여 노드 임베딩을 생성하는 프로젝트입니다.

#### 구현 내용
- 그래프 adjacency list 구성
- node2vec walk 생성
- `p`, `q` 파라미터에 따른 이동 가중치 계산
- walk 시퀀스를 기반으로 skip-gram 학습 수행
- embedding matrix 학습
- 특정 노드의 embedding 값 출력

#### 배운 점
- 그래프 구조를 벡터 표현으로 변환하는 임베딩 개념 이해
- node2vec의 `p`, `q` 파라미터가 탐색 성향에 미치는 영향 학습
- skip-gram이 단어뿐 아니라 그래프 노드 표현 학습에도 쓰일 수 있음을 이해
- representation learning 관점에서 그래프를 다루는 방법 경험

---

## Skills Demonstrated

- Python
- PySpark
- NumPy
- Graph Analysis
- PageRank
- Dead End / Spider Trap Analysis
- Neural Networks
- Forward Propagation
- Backpropagation
- Mini-batch Training
- Node Embedding
- Node2Vec
- Skip-Gram
- Representation Learning
- Algorithm Implementation
