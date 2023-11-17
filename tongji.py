import os
import sys
import statistics

logpath = 'log'
savepath = '/home/cyh/code/SSMGRL/results'
write_name_imdb = os.path.join(savepath, 'log_imdb.txt')
write_name_acm = os.path.join(savepath, 'log_acm.txt')
write_name_dblp = os.path.join(savepath, 'log_dblp.txt')
# write_name_amazon = os.path.join(savepath, 'summer_log_amazon1.txt')
# write_name_yelp = os.path.join(savepath, 'summer_log_yelp1.txt')
write_name_freebase = os.path.join(savepath, 'log_freebase.txt')
f_imdb = open(write_name_imdb, 'w')
f_acm = open(write_name_acm, 'w')
f_dblp = open(write_name_dblp, 'w')
# f_amazon = open(write_name_amazon, 'w')
# f_yelp = open(write_name_yelp, 'w')
f_freebase = open(write_name_freebase, 'w')
write_line = '\t'.join(['dataname','id','macF1', 'micF1','k1','kacc','knmi','sim','lambd0','lambd1', 'lambd2', 'w_loss1', 'w_loss2']) + '\n'
f_imdb.write(write_line)  #,'lambd3','lambd4', 'lambd5' , 'w_loss3', 'w_loss4', 'w_loss5'
f_acm.write(write_line)
f_dblp.write(write_line)
# f_amazon.write(write_line)
# f_yelp.write(write_line)
f_freebase.write(write_line)
Choose = {'acm':f_acm,'imdb':f_imdb,'dblp':f_dblp,'freebase':f_freebase} #,'ADNI2Cnew':f_oct, 'ADNIADMCI':f_arxiv
def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        dir_list = sorted(dir_list,key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        # print(dir_list)
        return dir_list
all_file_list = get_file_list(logpath)

lambd3 = 0.0  # !
lambd4 = 0.0  # !
lambd5 = 0.01  # !
w_loss3 = 0.0  # !
w_loss4 = 0.0
w_loss5 = 0.0

for filename in all_file_list:
    # if filename.find('.') == -1 or filename[-3:] == 'txt' :
    #     continue
    filenames = filename.split('_')
    dataname  = filenames[0] #!
    id = filenames[1] #!
    if len(filenames)>5:
        macF1 = filenames[2] #!
        micF1 = filenames[4]

    filepath = os.path.join(logpath,filename)
    # acc_list = []
    # for name in os.listdir(filepath):
    #     if name not in ['arg.txt'] and name[-3:] not in ['.py']:
    #         namelist = name.split('_')
    #         it = namelist[0]
    #         seed = namelist[1]
    #         acc = namelist[2]
    #         maxacc = namelist[3]
    #         acc_list.append(float(acc))
    #
    # std = statistics.pstdev(acc_list) #!
    std =0.1

    logname = os.path.join(filepath,'arg.txt')
    if(os.path.exists(logname)):
        logfile = open(logname, 'r')

        # w_loss1 = 0
        kacc = 0
        knmi = 0
        for line in logfile:
            if line.find('- lambd0:') == 0:
                lambd0 = line.split(':')[-1][1:-1]
            elif line.find('- lambd1:') == 0:
                lambd1 = line.split(':')[-1][1:-1]
            elif line.find('- lambd2:') == 0 :
                lambd2 = line.split(':')[-1][1:-1]
            elif line.find('- lambd3:') == 0:
                lambd3 = line.split(':')[-1][1:-1]
            elif line.find('- lambd4:') == 0:
                lambd4 = line.split(':')[-1][1:-1]
            elif line.find('- lambd5:') == 0:
                lambd5 = line.split(':')[-1][1:-1]
            elif line.find('- w_loss1:') == 0:
                w_loss1 = line.split(':')[-1][1:-1]
            elif line.find('- w_loss2:') == 0:
                w_loss2 = line.split(':')[-1][1:-1]
            # elif line.find('- w_recon1:') == 0:
            #     w_recon1 = line.split(':')[-1][1:-1]
            # elif line.find('- w_recon2:') == 0:
            #     w_recon2 = line.split(':')[-1][1:-1]
            elif line.find('- macro_f1s:') == 0:
                k1 = line.split('k1')[-1][1:-8]
                sim = line.split('similarity:[')[-1][1:30]
                kacc = line.split('kacc:')[-1][1:-1]
                knmi = line.split('knmi:')[-1][1:20]
            elif line.find('- kac') == 0:
                kacc = line.split('c')[-1][1:-8]
                knmi = line.split('knmi:')[-1][1:-8]
            # elif line.find('- w_loss3:') == 0:
            #     w_loss3 = line.split(':')[-1][1:-1]
            # elif line.find('- w_loss4:') == 0:
            #     w_loss4 = line.split(':')[-1][1:-1]
            # elif line.find('- w_loss5:') == 0:
            #     w_loss5 = line.split(':')[-1][1:-1]
            # elif line.find('- margin2:') == 0:
            #     margin2 = line.split(':')[-1][1:-1]
            #     # margin1 = line.split(':')[-1][1:-1]
            # elif line.find('- loss2:') == 0:
            #     loss2 = line.split(':')[-1][1:-1]
        if dataname in Choose:
            f = Choose[dataname]
            write_line = '\t'.join([dataname,id,macF1,micF1,k1,str(kacc),str(knmi),sim,lambd0, lambd1,lambd2,w_loss1,w_loss2]) + '\n'
            f.write(write_line)
f_imdb.close()
f_acm.close()
f_dblp.close()
# f_amazon.close()
# f_yelp.close()
f_freebase.close()
# f_oct.close()

