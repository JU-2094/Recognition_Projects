#!/usr/bin/env bash

mkdir train
mkdir test
mkdir valid

for i in {0..2}
do
    mkdir train/${i}
    mkdir test/${i}
    mkdir valid/${i}
done

