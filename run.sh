#!/bin/bash

if [ $# -eq 0 ]
then
  echo "==> at least one parameter should be passed into run script"
  exit
fi

model=$1

if [ $# -eq 2 ]
then
  if [ "$2" == "tmp" -o "$2" == "temp" ]
  then
    full_path="${model}_$2"
    config="./configs/${model}.jsonnet"
  else
    full_path="${model}_$2"
    config="./configs/${model}_$2.jsonnet"
  fi
else
  full_path="${model}"
  config="./configs/$model.jsonnet"
fi

serial_dir="`pwd`/models/model_$full_path"

echo "==> full path: $full_path"
echo "==> config file: $config"
echo "==> serialization dir: $serial_dir"

if [ -d $serial_dir ];then
  echo "==> remove existing serial folder"
  rm -rf $serial_dir
fi

if [ ! -f $config ];then
  echo "==> config file doesn't exist"
  exit
fi

allennlp train $config --serialization-dir $serial_dir
