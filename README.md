# level2_klue-nlp-01

### Background

> 문장 속에서 단어간 관계성을 파악하는 것은 의미나 의도를 해석함에 있어서 많은 도움을 줍니다. 요약된 정보를 활용하여 QA 시스템 구축을 하거나, 자연스러운 대화가 이어질 수 있도록 하는 바탕이 되기도 합니다. 
이처럼 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 Task입니다. 관계 추출은 지식 그래프를 구축하기 위한 핵심으로 구조화된 검색, 감정분석, 질의응답, 요약과 같은 다양한 NLP task의 기반이 됩니다.

### Target

> 이번 competition 에서는 한국어 문장과 subject_entity, object_entity가 주어졌을 때, entity 간의 관계를 추론하는 모델을 학습시키게 됩니다. 아래는 예시입니다.

```
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
subject_entity: 썬 마이크로시스템즈
object_entity: 오라클

relation: 단체:별칭 (org:alternate_names)
```

- input: sentence, subject_entity, object_entity
- output: pred_label, probs
