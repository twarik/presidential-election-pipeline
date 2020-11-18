#!/bin/sh

image_name=twarik/road_safety_pipeline_preprocessing
image_tag=latest

full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")"

docker build -t "${full_image_name}" .
docker push "$full_image_name"
