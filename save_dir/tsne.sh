#!/bin/bash

# 클래스 이름 배열 정의
class_names=("traffic light" "traffic sign" "vegetation" "terrain" "sky" "person" "rider" "car" "truck" "bus" "train" "motorcycle" "bicycle")

# 각 클래스 이름에 대해 스크립트 실행
for class in "${class_names[@]}"
do
    # 클래스 이름을 인수로 전달할 때 따옴표로 묶음
    python visualize_tsne.py --className "$class"
done
