# Relation_Extraction
🏅 Top 5% in Relation Extraction Task in Naver BoostCamp AI Tech

## 대회 설명
- 문장 속에서 단어간의 관계성을 예측한다.
```
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
subject_entity: 썬 마이크로시스템즈
object_entity: 오라클

relation: 단체:별칭 (org:alternate_names)
```

### 평가방법
- **`no_relation` class를 제외한 micro F1 score**
- **모든 class에 대한 AUPRC (Area Under the Precision-Recall Curve)**

### Dataset
- Train Dataset: 32470
- Test Datset: 7765

<details>
    <summary><b>자세히</b></summary>
    
- Example (data/train.csv)
  - `id`, `sentence`, `subject_entity`, `object_entity`, `label`, `source`로 구성
  - `sentence`: 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
  - `subject_entity`: {'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'}
  - `object_entity`: {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}
  - `label`: no_relation
  - `source`: wikipedia

- Relation Category
![1](https://user-images.githubusercontent.com/53552847/136692171-30942eec-fb83-4175-aa8d-13559ae2caf1.PNG)
  
</details>

### Hardware
- `GPU: Tesla V100 32GB`

## 실행
```
pip install -r requirements.txt

# Train
python train.py --rbert

# Inference
python inference.py
```

## Code
```
+- data (.gitignore) => Entity.ipynb를 활용하여 Typed Entity Marker Dataset 생성 
+- EDA
|   +- EDA.ipynb 
|   +- Entity.ipynb => Typed Entity Marker Dataset 생성
|   +- Round_Trip_Translation.ipynb => Data Augmentation with Poror
+- utils
|   +- loss.py
|   +- metrics.py
|   +- nlpdata_eda.py => Token Length 출력
|   +- get_cls.py => Backbone Model로부터 cls hidden state vector 출력
+- config.yaml
+- requirements.txt
+- dataset.py
+- train.py
+- inference.py
+- model.py
+- ensemble.py
```

## Core Strategy
- **Typed Entity Marker을 활용한 input text preprocessing**
  - Typed Entity Marker: [SUB-ORGANIZATION]아메리칸 리그[/SUB-ORGANIZATION]가 출범한 [OBJ-DATE]1901년[/OBJ-DATE] 당시 .426의 타율을 기록하였다.
  - Typed Entity Marker (punct): @*기관*아메리칸 리그@가 출범한 &^날짜^1901년& 당시 .426의 타율을 기록하였다.
- **RBERT** [(Paper Review 참고)](https://jjonhwa.github.io/booststudy/2022/02/13/booststudy-paper-RBERT/#3-methodology)
  - CLS hidden state vector, Entity1의 각 Token에 대한 Average hidden state vector, Entity2의 각 Token에 대한 Average hidden state vector
  - 각 hidden state vector를 Fully-Connected Layer + Activation 통과
  - 통과한 3개의 Vector를 Concatenate하여 하나의 vector로 만든 후 최종 분류 layer 통과
- **An Improved Baseline for Sentence-level Relation Extraction** [(Paper Review 참고)](https://jjonhwa.github.io/booststudy/2022/02/17/booststudy-paper-Improved_Baseline/#3-method)
  - Entity1의 start token에 대한 embedding vector, Entity2의 start token에 대한 embedding vector
  - 두 vector를 concatenate하여 하나의 vector를 만든 후 최종 분류 layer 통과 
- **Out of Fold Ensemble**
  - Startified KFold를 활용한 Ensemble 진행
  - Baseline + RBERT + Improved_Baseline Ensemble 

## 결과

||Custom Baseline|RBERT|Improved_Baseline|Ensemble public|Ensemble private|
|---|---|---|---|---|---|
|F1_Score|73.349|74.530|73.277|75.218|73.865|
|AUPRC|80.332|79.877|76.317|81.480|83.216|

## 팀원
- [Team - HappyFace](https://github.com/KR-HappyFace)
- [snoop2head](https://github.com/snoop2head)
- [jjonhwa](https://github.com/jjonhwa)
- [kimyeondu](https://github.com/kimyeondu)
- [hihellohowareyou](https://github.com/hihellohowareyou)
- [shawnhyeonsoo](https://github.com/shawnhyeonsoo)
- [Joonhong Kim](https://github.com/JoonHong-Kim)
- [ntommy11](https://github.com/ntommy11)
