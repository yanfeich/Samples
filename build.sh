#!/bin/bash
set -ex

BIN=`dirname ${0}`
ROOT=`cd ${BIN}; pwd`

SRC=${ROOT}/src
SAMPLES=`ls ${SRC}`
for sample in ${SAMPLES}
do
    cd ${SRC}/${sample}
    if [ -f ${SRC}/${sample}/Makefile ];then
        make
    fi
    cd -
done