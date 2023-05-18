import pandas as pd
import random
import os

from tqdm import tqdm

# train.csv 파일 읽어오기
reference_dataset = pd.read_csv("/opt/ml/level2_klue-nlp-01/dataset/dataset v.1.0/train.csv")

# 증강결과 저장용 데이터프레임 생성
augmented_dataset = pd.DataFrame(columns=reference_dataset.columns)

# 증강할 label 설정
labels = ["no_relation"]

# 각 type에 대한 교체할 entity 후보 사전 만들기
entity_types = ["DAT", "POH", "LOC", "ORG", "NOH", "PER"]
candidates = dict()
for entity in entity_types:
    if entity == "PER" or entity == "ORG":
        temp = reference_dataset[
            (reference_dataset["subject_entity"].str.contains(entity))
        ]  # sub_entity에는 'PER', 'ORG' type만 존재하므로
        candidates[entity] = temp.append(
            reference_dataset[(reference_dataset["object_entity"].str.contains(entity))],
            ignore_index=True,
        )
    else:
        candidates[entity] = reference_dataset[
            (reference_dataset["object_entity"].str.contains(entity))
        ]


# 각 레이블에 대해 데이터 증강
for label in tqdm(labels):
    # 해당 레이블의 데이터 개수
    label_count = reference_dataset[reference_dataset["label"] == label].shape[0]

    # 증강할 개수 설정(no_realtion 개수 만큼 or 임의로 설정)
    ## no_realtion 개수 만큼 나머지 데이터들을 증강
    # no_relation_count = reference_dataset[reference_dataset["label"] == "no_relation"].shape[0]
    # target_augmentation_count = no_relation_count - label_count

    ## 데이터 증강할 개수 임의로 설정 (no_realtion 개수를 기준으로 하지 않을 때)
    target_augmentation_count = 1

    # 현재 label에 대해 현재까지 생성된 데이터의 수 0으로 초기화
    generated_count = 0

    # 증강된 데이터, 원본 데이터 포함하여 같은 sentence가 존재하지 않도록 처리하여 증강
    while True:
        # 현재 증강해야될 개수 계산
        augmentation_count = max(0, target_augmentation_count - generated_count)
        if augmentation_count == 0:  # 증강해야될 데이터의 개수
            break

        # 해당 레이블의 데이터 샘플링
        label_data = reference_dataset[
            reference_dataset["label"] == label
        ]  # 해당 label에 해당하는 모든 데이터 추출

        # 증강된 데이터 생성
        augmented_data = label_data.sample(
            augmentation_count, replace=True
        )  # 증강하고자하는 데이터만큼 샘플링하여 추출

        for _, row in augmented_data.iterrows():
            # 해당 row의 레이블
            label = row["label"]
            sentence = row["sentence"]
            subject_entity = eval(row["subject_entity"])
            subject_entity_type = subject_entity["type"]

            # object_entity 증강
            object_entity = eval(row["object_entity"])
            object_entity_type = object_entity["type"]

            # 'POH' 또는 'NOH' 타입의 경우 subject_entity만 대체
            if object_entity_type in ["POH", "NOH"]:
                subject_candidates = candidates[subject_entity_type]
                if not subject_candidates.empty:
                    subject_replacement = subject_candidates.sample(n=1)
                    new_subject_entity = eval(subject_replacement["subject_entity"].iloc[0])

                    # 생성된 데이터 중 이미 기존에 존재하는 데이터가 존재하는지 확인
                    # 'sentence' 열에서 동일한 문자열을 가진 행이 있거나 기존 reference와 생성될 데이터의 sentence가 같다면 생성하지 않음
                    new_sentence = row["sentence"].replace(
                        subject_entity["word"], new_subject_entity["word"]
                    )

                    is_existing = augmented_dataset["sentence"].eq(new_sentence).any()
                    if new_sentence == sentence or is_existing:
                        continue

                    subject_word = subject_entity["word"]  # 기존 row의 subject_entity word
                    subject_start_idx = subject_entity["start_idx"]
                    subject_end_idx = subject_entity["end_idx"]
                    subject_length_diff = len(new_subject_entity["word"]) - len(subject_word)

                    #  entity start_idx, end_idx 업데이트
                    # subject_entity의 시작 인덱스와 끝 인덱스 계산
                    subject_entity_count = sentence[:subject_start_idx].count(subject_word) + 1
                    new_subject_entity_start_idx = -1
                    new_subject_entity_end_idx = 0

                    count = 0
                    for i in range(len(new_sentence)):
                        if (
                            new_sentence[i : i + len(new_subject_entity["word"])]
                            == new_subject_entity["word"]
                        ):
                            count += 1
                            if count == subject_entity_count:
                                new_subject_entity_start_idx = i
                                break
                    new_subject_entity_end_idx = (
                        new_subject_entity_start_idx + len(new_subject_entity["word"]) - 1
                    )

                    subject_entity["word"] = new_subject_entity["word"]
                    subject_entity["start_idx"] = new_subject_entity_start_idx
                    subject_entity["end_idx"] = new_subject_entity_end_idx

            # 다른 타입의 경우 subject_entity와 object_entity 모두 대체
            else:
                subject_candidates = candidates[subject_entity_type]
                object_candidates = candidates[object_entity_type]
                if (
                    not subject_candidates.empty and not object_candidates.empty
                ):  # 둘 다 후보들이 한 개 이상 뽑혔다면
                    subject_replacement = subject_candidates.sample(
                        n=1
                    )  # 후보 row들(같은 subject_entity type 갖은)중 한 개 선택
                    object_replacement = object_candidates.sample(
                        n=1
                    )  # 후보 row들(같은 object_entity type 갖은)들 중 한 개 선택
                    new_subject_entity = eval(
                        subject_replacement["subject_entity"].iloc[0]
                    )  # 대체하기 위해 가져온 row의 subject_entity 추출
                    new_object_entity = eval(
                        object_replacement["object_entity"].iloc[0]
                    )  # 대체하기 위해 가져온 row의 object_entity 추출
                    new_sentence = row["sentence"].replace(
                        subject_entity["word"], new_subject_entity["word"]
                    )  # 새로운 word로 sentence 업데이트
                    new_sentence = new_sentence.replace(
                        object_entity["word"], new_object_entity["word"]
                    )  # 새로운 word로 sentence 업데이트

                    # 생성된 데이터 중 이미 기존에 존재하는 데이터가 존재하는지 확인
                    # 'sentence' 열에서 동일한 문자열을 가진 행이 있거나 기존 reference와 생성될 데이터의 sentence가 같다면 생성하지 않음
                    is_existing = augmented_dataset["sentence"].eq(new_sentence).any()
                    if new_sentence == sentence or is_existing:
                        continue

                    subject_word = subject_entity["word"]  # 기존 row의 subject_entity word
                    subject_start_idx = subject_entity["start_idx"]
                    subject_end_idx = subject_entity["end_idx"]
                    subject_length_diff = len(new_subject_entity["word"]) - len(subject_word)

                    object_word = object_entity["word"]
                    object_start_idx = object_entity["start_idx"]
                    object_end_idx = object_entity["end_idx"]  # reference object의 end_idx
                    object_length_diff = len(new_object_entity["word"]) - len(object_word)

                    # entity start_idx, end_idx 업데이트
                    # object_entity의 시작 인덱스와 끝 인덱스 계산
                    object_entity_count = sentence[:object_start_idx].count(object_word) + 1
                    new_object_entity_start_idx = -1
                    new_object_entity_end_idx = 0

                    count = 0
                    for i in range(len(new_sentence)):
                        if (
                            new_sentence[i : i + len(new_object_entity["word"])]
                            == new_object_entity["word"]
                        ):
                            count += 1
                            if count == object_entity_count:
                                new_object_entity_start_idx = i
                                break
                    new_object_entity_end_idx = (
                        new_object_entity_start_idx + len(new_object_entity["word"]) - 1
                    )

                    # subject_entity의 시작 인덱스와 끝 인덱스 계산
                    subject_entity_count = sentence[:subject_start_idx].count(subject_word) + 1
                    new_subject_entity_start_idx = -1
                    new_subject_entity_end_idx = 0

                    count = 0
                    for i in range(len(new_sentence)):
                        if (
                            new_sentence[i : i + len(new_subject_entity["word"])]
                            == new_subject_entity["word"]
                        ):
                            count += 1
                            if count == subject_entity_count:
                                new_subject_entity_start_idx = i
                                break
                    new_subject_entity_end_idx = (
                        new_subject_entity_start_idx + len(new_subject_entity["word"]) - 1
                    )

                    subject_entity["word"] = new_subject_entity["word"]
                    subject_entity["start_idx"] = new_subject_entity_start_idx
                    subject_entity["end_idx"] = new_subject_entity_end_idx
                    object_entity["word"] = new_object_entity["word"]
                    object_entity["start_idx"] = new_object_entity_start_idx
                    object_entity["end_idx"] = new_object_entity_end_idx

            # 변경된 데이터를 새로운 row로 추가
            augmented_row = row.copy()
            augmented_row["sentence"] = new_sentence
            augmented_row["subject_entity"] = str(subject_entity)
            augmented_row["object_entity"] = str(object_entity)
            augmented_dataset = augmented_dataset.append(augmented_row, ignore_index=True)
            generated_count += 1

# 모든 데이터가 증강된 후 랜덤으로 shuffle
augmented_dataset = augmented_dataset.sample(frac=1, random_state=42)

# 기존 데이터와 concat (id를 기존 데이터프레임의 가장 높은 수부터 차례로 증가하며 할당)
max_id_number = reference_dataset["id"].max() + 1
augmented_dataset["id"] = range(max_id_number, max_id_number + len(augmented_dataset))

# 생성된 데이터셋 저장 경로 설정
save_dir = "/opt/ml/level2_klue-nlp-01/dataset/augmented_dataset"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 최종적으로 만들어진 데이터셋 저장
final_dataset = reference_dataset.append(augmented_dataset, ignore_index=True)
final_dataset.to_csv(os.path.join(save_dir, "augmented_train______.csv"), index=False)
