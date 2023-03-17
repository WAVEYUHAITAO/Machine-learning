import random
import math
import numpy as np
import pandas as pd
import re

def decoder(x1):
    result1=x1/1024*40-20
    return result1

def encoder(x):
    result=(x+20)/40*1024
    return bin(result)

def creat_person(num):
    family=np.zeros((num,7))
    for i in range(num):
        for j in range(3):
            idx = random.randint(0, 1023)
            family[i,j]=idx
            print("individual"+str(i)+"x"+str(j)+" binary_code:"+bin(idx))
            family[i,j+3]=decoder(idx)
            print("individual" + str(i) + "x" + str(j) + " value: " + str(family[i,j+2]))
        family[i,6]=fitness(family[i,0:6])
        print("individual" + str(i) +  " fitness: " + str(family[i, 6]))
    return family
def fitness(family_row):
    x1=family_row[3]
    x2=family_row[4]
    x3=family_row[5]
    fit=-20*math.exp(-0.2*math.sqrt((1/3)*(x1*x1+x2*x2+x3*x3)))-math.exp(1/3*(math.cos(2*math.pi*x1)+math.cos(2*math.pi*x2)+math.cos(2*math.pi*x3)))+20+math.exp(1)

    return fit
def fitness_new(x1,x2,x3):
    fit=-20*math.exp(-0.2*math.sqrt((1/3)*(x1*x1+x2*x2+x3*x3)))-math.exp(1/3*(math.cos(2*math.pi*x1)+math.cos(2*math.pi*x2)+math.cos(2*math.pi*x3)))+20+math.exp(1)

    return fit
def string_to_bin(binary,len=10):
    result=np.zeros((1,len))
    # print(binary)
    for i in range(1,len+1):
        if binary[-i]!='b':
            result[0,len-i]=float(binary[-i])
        else:
            break
    return result


def data_to_barrier(data,len=10):
    data_list=data.copy()
    for i in range(1,len):
        data_list[i]=data_list[i-1]+data_list[i]
    return data_list

def data_to_barrier_new(data,len=10):
    data_list=data.copy()
    for i in range(1,len):
        data_list[i]=data_list[i-1]+data_list[i]
    return data_list

def percent_calculate(column):
    new_column=-column+25
    total=new_column.sum()
    final_column=new_column/total*1000
    return final_column
def percent_calculate_new(column):
    new_column=-column+25
    total=new_column.sum()
    final_column=new_column/total*1000
    return final_column
def select_father_and_mother_index(seeds,whole):
    for i in range(whole.shape[0]):
        if seeds<whole[i,8]:
            return i
def select_parent_index(seeds,barrier):
    for i in range(len(barrier)):
        if seeds<barrier[i]:
            return i
def creat_new(seed_begin,seeds,father,mather):
    seed=seed_begin

    for i in range(10):
        seed=seed+1

def find_seed(num,seeds):
    if num>=200:
        num=num%200
    seed=seeds[num]
    return num+1,seed
def get_gen_str(gen_all):
    str_result=""
    for i in range(len(gen_all)):
        gen=string_to_bin(bin(int(gen_all[i]))).flatten()
        for j in gen:
            stt=str(int(j))
            str_result="".join([str_result,stt])
    return str_result
def calculate_children(mather,father,seed_index,rand_seeds):
    gen_mask=""
    children_gen=""
    seed_record=""
    for i in range(30):
        seed_index,seed=find_seed(seed_index,rand_seeds)
        if seed<=450:
            seed_record = seed_record + "" + str(seed) + ","
            children_gen=children_gen+mather[i]
            gen_mask=gen_mask+"0"
        elif seed<=900:
            seed_record = seed_record + "" + str(seed) + ","
            children_gen=children_gen+father[i]
            gen_mask=gen_mask+"1"
        else:
            gen_mask=gen_mask+"2"
            seed_index,muta=find_seed(seed_index,rand_seeds)
            seed_record = seed_record + "" + str(seed) + "and"
            seed_record=seed_record+str(muta)+" for muta"+","
            if muta>500:
                children_gen=children_gen+"1"
            else:
                children_gen=children_gen+"0"
    # length=len(children_gen)
    # print(length)
    return seed_index,children_gen,seed_record,gen_mask
def analyze_gen(gen_code):
    if len(gen_code)!=30:
        print("error code!")
    gen2 = re.findall(r'.{10}', gen_code)
    binary_x1=gen2[0]
    binary_x2=gen2[1]
    binary_x3=gen2[2]
    x1=int(binary_x1,2)
    x2=int(binary_x2,2)
    x3=int(binary_x3,2)
    x1_value=decoder(x1)
    x2_value=decoder(x2)
    x3_value=decoder(x3)
    fit=fitness_new(x1_value,x2_value,x3_value)
    return x1_value,x2_value,x3_value,binary_x1,binary_x2,binary_x3,x1,x2,x3,fit

if __name__=="__main__":
    # np.save("haha.npy",vis)
    # test=np.array([1,0,1,0,1,0])
    # # print(fitness(test))
    # fitness_g1=np.zeros((num,1))
    # random=np.loadtxt('1.txt',dtype=np.int32)
    # random=random.reshape(-1,5)
    # np.save("rand",random)
    # pd.DataFrame(columns=['A', 'B', 'C'], index=[0, 1, 2])
    matric_number=60
    user_name="yht"

    # matric_number=26
    # user_name="wdh"

    seed_index=matric_number-1
    iteration=3
    iteration=iteration+1

    rand_seeds=np.load("E:\\Repositories\\EE6227_GAML\\Homework\\rand.npy").flatten(order="F")
    num=10
    initial=creat_person(num)# in this function we get the initial and 10-bits code
    # print(initial)
    initial=np.c_[initial,percent_calculate(initial[:,6])]
    initial=np.c_[initial,data_to_barrier(initial[:,7],len=10)]


    children_table=pd.DataFrame(columns=['father_index', 'seed_choose_father', 'father_gen_30bits',"mother_index","seed_choose_mather","mother_gen_30bits","seed_to_decide_gen_str","children_gen_mask","children_gen_30bits","x1_binary","x1_coding","x1_value","x2_binary","x2_coding","x2_value","x3_binary","x3_coding","x3_value","fuction_value","fitness%。","barrier"], index=range(10*iteration))
#meaningless just put initial in to children_table
    for i in range(10):
        children_gen=get_gen_str(initial[i,0:3])
        children_table.loc[i, "children_gen_30bits"] = children_gen
        x1, x2, x3, binary_x1, binary_x2, binary_x3, x1_coding, x2_coding, x3_coding, fitness_value = analyze_gen(children_gen)
        children_table.loc[i, "x1_value"] = x1
        children_table.loc[i, "x2_value"] = x2
        children_table.loc[i, "x3_value"] = x3
        children_table.loc[i, "x1_binary"] = binary_x1
        children_table.loc[i, "x2_binary"] = binary_x2
        children_table.loc[i, "x3_binary"] = binary_x3
        children_table.loc[i, "x1_coding"] = x1_coding
        children_table.loc[i, "x2_coding"] = x2_coding
        children_table.loc[i, "x3_coding"] = x3_coding
        children_table.loc[i, "fuction_value"] = initial[i,6]
        children_table.loc[i, "fitness%。"] =initial[i,7]
        children_table.loc[i, "barrier"] = initial[i,8]

    for i in range(10,20):
        seed_index,seed_father=find_seed(seed_index,rand_seeds)
        print(f"father seed index:{seed_index}")
        print(f"seed_father:{seed_father}")
        seed_index,seed_mather=find_seed(seed_index,rand_seeds)#a=get father or mother index
        print(f"mather seed index:{seed_index}")
        print(f"seed_mather:{seed_mather}")
        #set a table for datasaving
        father_index = select_father_and_mother_index(seed_father, initial)
        father_gen = get_gen_str(initial[father_index, 0:3])
        mather_index = select_father_and_mother_index(seed_mather, initial)
        mather_gen = get_gen_str(initial[mather_index, 0:3])
        children_table.loc[i,"seed_choose_father"]=seed_father
        children_table.loc[i,"father_index"]=father_index
        children_table.loc[i,"father_gen_30bits"]=father_gen
        children_table.loc[i, "seed_choose_mather"] = seed_mather
        children_table.loc[i, "mother_index"] = mather_index
        children_table.loc[i, "mother_gen_30bits"] = mather_gen
        seed_index,children_gen,seeds_gen,gen_mask=calculate_children(mather_gen,father_gen,seed_index,rand_seeds)
        children_table.loc[i, "children_gen_30bits"] = children_gen
        children_table.loc[i, "seed_to_decide_gen_str"] = seeds_gen
        children_table.loc[i,"children_gen_mask"]=gen_mask
        x1,x2,x3,binary_x1,binary_x2,binary_x3,x1_coding,x2_coding,x3_coding,fitness_value=analyze_gen(children_gen)
        children_table.loc[i,"x1_value"]=x1
        children_table.loc[i,"x2_value"]=x2
        children_table.loc[i,"x3_value"]=x3
        children_table.loc[i,"x1_binary"]=binary_x1
        children_table.loc[i,"x2_binary"]=binary_x2
        children_table.loc[i,"x3_binary"]=binary_x3
        children_table.loc[i,"x1_coding"]=x1_coding
        children_table.loc[i,"x2_coding"]=x2_coding
        children_table.loc[i,"x3_coding"]=x3_coding
        children_table.loc[i,"fuction_value"] = fitness_value
        print("First Generation Children "+str(i+1)+" Completed!")
        #gen_mask 0:mother 1:father 2:muta
    children_table.loc[10:19,"fitness%。"]=percent_calculate(children_table.loc[10:19,"fuction_value"].to_numpy())
    children_table.loc[10:19,"barrier"]=data_to_barrier(children_table.loc[10:19,"fitness%。"].to_numpy(),10)
    for j in range(2,iteration):
        for i in range(10*j,10*j+10):
            seed_index, seed_father = find_seed(seed_index, rand_seeds)
            seed_index, seed_mather = find_seed(seed_index, rand_seeds)  # a=get father or mother index
            father_index = select_parent_index(seed_father, children_table.loc[10*(j-1):10*j-1,"barrier"].to_numpy())+10*j-10
            father_gen = children_table.loc[father_index, "children_gen_30bits"]
            mather_index = select_parent_index(seed_mather, children_table.loc[10*(j-1):10*j-1,"barrier"].to_numpy())+10*j-10
            mather_gen = children_table.loc[mather_index, "children_gen_30bits"]
            children_table.loc[i, "seed_choose_father"] = seed_father
            children_table.loc[i, "father_index"] = father_index
            children_table.loc[i, "father_gen_30bits"] = father_gen
            children_table.loc[i, "seed_choose_mather"] = seed_mather
            children_table.loc[i, "mother_index"] = mather_index
            children_table.loc[i, "mother_gen_30bits"] = mather_gen
            seed_index, children_gen, seeds_gen, gen_mask = calculate_children(mather_gen, father_gen, seed_index,rand_seeds)
            children_table.loc[i, "children_gen_30bits"] = children_gen
            children_table.loc[i, "seed_to_decide_gen_str"] = seeds_gen
            children_table.loc[i, "children_gen_mask"] = gen_mask
            x1, x2, x3, binary_x1, binary_x2, binary_x3, x1_coding, x2_coding, x3_coding, fitness_value = analyze_gen(children_gen)
            children_table.loc[i, "x1_value"] = x1
            children_table.loc[i, "x2_value"] = x2
            children_table.loc[i, "x3_value"] = x3
            children_table.loc[i, "x1_binary"] = binary_x1
            children_table.loc[i, "x2_binary"] = binary_x2
            children_table.loc[i, "x3_binary"] = binary_x3
            children_table.loc[i, "x1_coding"] = x1_coding
            children_table.loc[i, "x2_coding"] = x2_coding
            children_table.loc[i, "x3_coding"] = x3_coding
            children_table.loc[i, "fuction_value"] = fitness_value
            print(" Generation "+str(j+1)+" Children " + str(i-10*j+1) + " Completed!")
        children_table.loc[10*j:10*j+9, "fitness%。"] = percent_calculate_new(children_table.loc[10*j:10*j+9, "fuction_value"].to_numpy())
        children_table.loc[10*j:10*j+9, "barrier"] = data_to_barrier_new(children_table.loc[10*j:10*j+9, "fitness%。"].to_numpy(), 10)

    # index=select_father_and_mother_index(300,initial)
    # colunmns=range(1,5)
    # children_list=pd.DataFrame(columns=[], index=[0, 1, 2]
    filename=user_name+"ga_ca_children_table.xls"
    children_table.to_excel(filename)
    print("finish_save")





