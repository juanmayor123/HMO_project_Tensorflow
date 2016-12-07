  # for i in {370..712}
  for ((i=370; i<=712; i++)); do
     echo $i  
     python loto_code_DNN.py test_sub4.csv 0 temp_loto_train.csv 712 $i temp_loto_test.csv
     sed '1d' temp_loto_train.csv > tmpfile.txt ; mv tmpfile.txt temp_loto_train.csv
     python fully_connected_resp.py temp_loto_train.csv temp_loto_test.csv cs_loto_sub4.csv 712 0 1
     head -n 1 cs_loto_sub4.csv >> tmp_file_lab.txt ; cp tmp_file_lab.txt cs_loto_sub4.csv
   done