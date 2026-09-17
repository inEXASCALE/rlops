#!/bin/bash

SRC="results_to_statistics"
FILES=("calculate_average_precisions1.py" "calculate_average_precisions2.py")
TARGETS=("results_random_dense1" "results_random_dense2" "results_random_dense3" "results_random_dense4")

# 拷贝文件
for DIR in "${TARGETS[@]}"; do
    echo "📁 正在拷贝文件到 $DIR ..."
    for FILE in "${FILES[@]}"; do
        cp "${SRC}/${FILE}" "${DIR}/"
    done
done

echo "指定 Python 文件已成功拷贝！"
