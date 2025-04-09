make 
# cd ../script
task_count=0
querysets=("uniform1000")
model_dir="/your/model/dir/" 

dataid="sun"
model_name="/your/model/name"
numactl --cpunodebind=0 --membind=0 ./main "${dataid}" "32" "64" "100" "${model_dir}" "${model_name}" "${task_count}" "${#querysets[@]}" "${querysets[@]}" 
