
data_dir='/your/path/'
model_dir="/your/path/"
queryid='uniform1000'
task_count='DEFL_IVF'
nlist='1024'

dataid='glove'
model_name='(60_512_512)glove_abs.cpt'
r='1.05'
../build/main "${data_dir}" "${dataid}" "${queryid}" "${model_dir}" "${model_name}" "$task_count" "$r" "${nlist}" 