#!/bin/bash

CURRENT_DIR=$(
    cd "$(dirname "$0")"
    pwd
)

LOG_DIR=${CURRENT_DIR}/../../logs
SUBDIALECT_DIR=/root/data/KeSpeech/KeSpeech/Subdialects

cd ${CURRENT_DIR}/../tools
for subdialect in $(ls ${SUBDIALECT_DIR}); do
    echo ${subdialect}
    HYP_FILE=${LOG_DIR}/recognition_result/${subdialect}.log
    REF_FILE=${SUBDIALECT_DIR}/${subdialect}/test_phase1/text
    echo ${REF_FILE}
    echo ${HYP_FILE}

    python compute-cer.py ${REF_FILE} ${HYP_FILE} >${LOG_DIR}/cer/${subdialect}.cer
done
