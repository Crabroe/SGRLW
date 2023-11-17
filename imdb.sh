#!/usr/bin/env bash
#declare lam0=0.0001
#declare lam1=0.0001
#declare lam2=0.0001
#declare w_1=0.01
#declare w_2=0.01
#lam0=1/10000
#lam1=0.0001
#lam2=0.0001
w_1=(0.01 0.1 1 10)
w_2=(0.01 0.1 1 10)
#w_recon1=(0.01 0.1 1 10 100)
#w_recon2=(0.01 0.1 1 10 100)
num=0
lam0=(0.0001 0.0005 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.05)
lam1=(0.0001 0.0005 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.05)
lam2=(0.0001 0.0005 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.05)

for(( i=0;i<${#lam0[@]};i++)) do
  for(( j=0;j<${#lam1[@]};j++)) do
    for(( k=0;k<${#lam2[@]};k++)) do
      for(( m=0;m<${#w_1[@]};m++)) do
        for(( n=0;n<${#w_2[@]};n++)) do
#${#array[@]}获取数组长度用于循环
((num+=1))
echo num: $num
if [ $num -ge 2056 ];
then
    /home/myj/anaconda3/envs/SuperGAT1.6/bin/python  main_imdb.py --lambd0 ${lam0[i]} --lambd1 ${lam1[j]} --lambd2 ${lam2[k]}  --w_loss1 ${w_1[m]} --w_loss2 ${w_2[n]}
#statements
fi
    done
    done
    done
    done
done
#python main_dblp.py --w_loss1 1 --w_loss2 10 --w_loss3 10  --w_loss11 10 --w_loss4 10































