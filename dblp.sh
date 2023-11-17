#!/usr/bin/env bash
#declare lam0=0.0001
#declare lam1=0.0001
#declare lam2=0.0001
#declare w_1=0.01
#declare w_2=0.01
#lam0=1/10000
#lam1=0.0001
#lam2=0.0001
w_1=(0.1 1 10)
w_2=(0.1 1 10)
w_3=(0.1 1 10)
w_4=(0.1 1 10)
w_5=(0.1 1 10)
#w_recon1=(0.01 0.1 1 10 100)
#w_recon2=(0.01 0.1 1 10 100)
num=0
lam0=(0.0001 0.0005 0.001  0.005 0.01 0.05)
lam1=(0.0001 0.0005 0.001  0.005 0.01 0.05)
lam2=(0.0001 0.0005 0.001  0.005 0.01 0.05)
lam3=(0.0001 0.0005 0.001  0.005 0.01 0.05)
lam4=(0.0001 0.0005 0.001  0.005 0.01 0.05)
lam5=(0.0001 0.0005 0.001  0.005 0.01 0.05)
#statements

for(( i=0;i<${#lam0[@]};i++)) do
  for(( j=0;j<${#lam1[@]};j++)) do
    for(( k=0;k<${#lam2[@]};k++)) do
      for(( l=0;l<${#lam3[@]};l++)) do
        for(( o=0;o<${#lam4[@]};o++)) do
          for(( p=0;p<${#lam5[@]};p++)) do
            for(( m=0;m<${#w_1[@]};m++)) do
              for(( n=0;n<${#w_2[@]};n++)) do
                for(( q=0;q<${#w_3[@]};q++)) do
                  for(( r=0;r<${#w_4[@]};r++)) do
                    for(( s=0;s<${#w_5[@]};s++)) do
#${#array[@]}获取数组长度用于循环
((num+=1))
echo num: $num
if [ $num -ge 1214 ];
then
    /home/myj/anaconda3/envs/SuperGAT1.6/bin/python  main_dblp.py --lambd0 ${lam0[i]} --lambd1 ${lam1[j]} --lambd2 ${lam2[k]} --lambd3 ${lam3[l]} --lambd4 ${lam4[o]} --lambd5 ${lam5[p]} --w_loss1 ${w_1[m]} --w_loss2 ${w_2[n]} --w_loss3 ${w_3[q]} --w_loss4 ${w_4[r]} --w_loss5 ${w_5[s]}
fi
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
done