# Relation_Extraction
π₯ Top 3 in Relation Extraction Task in Naver BoostCamp AI Tech

## λν μ€λͺ
- λ¬Έμ₯ μμμ λ¨μ΄κ°μ κ΄κ³μ±μ μμΈ‘νλ€.
```
sentence: μ€λΌν΄(κ΅¬ μ¬ λ§μ΄ν¬λ‘μμ€νμ¦)μμ μ κ³΅νλ μλ° κ°μ λ¨Έμ  λ§κ³ λ κ° μ΄μ μ²΄μ  κ°λ°μ¬κ° μ κ³΅νλ μλ° κ°μ λ¨Έμ  λ° μ€νμμ€λ‘ κ°λ°λ κ΅¬ν λ²μ μ μ¨μ ν μλ° VMλ μμΌλ©°, GNUμ GCJλ μνμΉ μννΈμ¨μ΄ μ¬λ¨(ASF: Apache Software Foundation)μ νλͺ¨λ(Harmony)μ κ°μ μμ§μ μμ νμ§ μμ§λ§ μ§μμ μΈ μ€ν μμ€ μλ° κ°μ λ¨Έμ λ μ‘΄μ¬νλ€.
subject_entity: μ¬ λ§μ΄ν¬λ‘μμ€νμ¦
object_entity: μ€λΌν΄

relation: λ¨μ²΄:λ³μΉ­ (org:alternate_names)
```

### νκ°λ°©λ²
- **`no_relation` classλ₯Ό μ μΈν micro F1 score**
- **λͺ¨λ  classμ λν AUPRC (Area Under the Precision-Recall Curve)**

### Dataset
- Train Dataset: 32470
- Test Datset: 7765

<details>
    <summary><b>μμΈν</b></summary>
    
- Example (data/train.csv)
  - `id`, `sentence`, `subject_entity`, `object_entity`, `label`, `source`λ‘ κ΅¬μ±
  - `sentence`: μ‘°μ§ ν΄λ¦¬μ¨μ΄ μ°κ³  λΉνμ¦κ° 1969λ μ¨λ² γAbbey Roadγμ λ΄μ λΈλλ€.
  - `subject_entity`: {'word': 'μ‘°μ§ ν΄λ¦¬μ¨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'}
  - `object_entity`: {'word': 'λΉνμ¦', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}
  - `label`: no_relation
  - `source`: wikipedia

- Relation Category
![1](https://user-images.githubusercontent.com/53552847/136692171-30942eec-fb83-4175-aa8d-13559ae2caf1.PNG)
  
</details>

### Hardware
- `GPU: Tesla V100 32GB`

## μ€ν
```
pip install -r requirements.txt

# Train
python train.py --mode 'rbert'

# Inference
python inference.py --mode 'rbert'
```

## Code
```
+- data (.gitignore) => Entity.ipynbλ₯Ό νμ©νμ¬ Typed Entity Marker Dataset μμ± 
+- EDA
|   +- EDA.ipynb 
|   +- Entity.ipynb => Typed Entity Marker Dataset μμ±
|   +- Round_Trip_Translation.ipynb => Data Augmentation with Poror
+- utils
|   +- loss.py
|   +- metrics.py
|   +- nlpdata_eda.py => Token Length μΆλ ₯
|   +- get_cls.py => Backbone Modelλ‘λΆν° cls hidden state vector μΆλ ₯
+- config.yaml
+- requirements.txt
+- dataset.py
+- train.py
+- inference.py
+- model.py
+- ensemble.py
```

## Core Strategy
- **Typed Entity Markerμ νμ©ν input text preprocessing**
  - Typed Entity Marker: `[SUB-ORGANIZATION]μλ©λ¦¬μΉΈ λ¦¬κ·Έ[/SUB-ORGANIZATION]κ° μΆλ²ν [OBJ-DATE]1901λ[/OBJ-DATE] λΉμ .426μ νμ¨μ κΈ°λ‘νμλ€.`
  - Typed Entity Marker (punct): `@*κΈ°κ΄*μλ©λ¦¬μΉΈ λ¦¬κ·Έ@κ° μΆλ²ν &^λ μ§^1901λ& λΉμ .426μ νμ¨μ κΈ°λ‘νμλ€.`
- **RBERT** [(Paper Review μ°Έκ³ )](https://jjonhwa.github.io/booststudy/2022/02/13/booststudy-paper-RBERT/#3-methodology)
  - CLS hidden state vector, Entity1μ κ° Tokenμ λν Average hidden state vector, Entity2μ κ° Tokenμ λν Average hidden state vector
  - κ° hidden state vectorλ₯Ό Fully-Connected Layer + Activation ν΅κ³Ό
  - ν΅κ³Όν 3κ°μ Vectorλ₯Ό Concatenateνμ¬ νλμ vectorλ‘ λ§λ  ν μ΅μ’ λΆλ₯ layer ν΅κ³Ό
- **An Improved Baseline for Sentence-level Relation Extraction** [(Paper Review μ°Έκ³ )](https://jjonhwa.github.io/booststudy/2022/02/17/booststudy-paper-Improved_Baseline/#3-method)
  - Entity1μ start tokenμ λν embedding vector, Entity2μ start tokenμ λν embedding vector
  - λ vectorλ₯Ό concatenateνμ¬ νλμ vectorλ₯Ό λ§λ  ν μ΅μ’ λΆλ₯ layer ν΅κ³Ό 
- **Out of Fold Ensemble**
  - Stratified KFoldλ₯Ό νμ©ν Ensemble μ§ν
  - Baseline + RBERT + Improved_Baseline Ensemble 

## κ²°κ³Ό

||Custom Baseline|RBERT|Improved_Baseline|Ensemble public|Ensemble private|
|---|---|---|---|---|---|
|F1_Score|73.349|74.530|73.277|75.218|73.865|
|AUPRC|80.332|79.877|76.317|81.480|83.216|

## νμ
- [Team - HappyFace](https://github.com/KR-HappyFace)
- [snoop2head](https://github.com/snoop2head)
- [jjonhwa](https://github.com/jjonhwa)
- [kimyeondu](https://github.com/kimyeondu)
- [hihellohowareyou](https://github.com/hihellohowareyou)
- [shawnhyeonsoo](https://github.com/shawnhyeonsoo)
- [Joonhong Kim](https://github.com/JoonHong-Kim)
- [ntommy11](https://github.com/ntommy11)
