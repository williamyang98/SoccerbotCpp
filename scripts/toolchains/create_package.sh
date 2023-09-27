#!/bin/sh
build_dir="build"
output_dir="soccerbotcpp_build"

rm -rf ${output_dir}
mkdir -p ${output_dir}

cp ${build_dir}/*.exe ${output_dir}/
cp ${build_dir}/*.dll ${output_dir}/
cp *.ini ${output_dir}/

cp -rf models/ ${output_dir}/
cp README.md ${output_dir}/
