eos_dir=/store/user/ammitra/NNCPE/
files=$(eosls $eos_dir | grep step2_DIGI_L1_DIGI2RAW_HLT )
file_string=""
for file in $files; do
    file_string=$file_string$eosprefix$eos_dir/$file,
done
file_string=${file_string%?}
echo $file_string 
edmCopyPickMerge outputFile="merged.root" inputFiles=$file_string
