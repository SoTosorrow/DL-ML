import operator;

# 字典中排序
def sortDict():
    dic={'a':1,'f':2,'c':3,'h':0};
    # 函数原型：sorted(dic,value,reverse)
    # 按字典中的键进行升序排序
    print("按键进行升序排序结果为:",\
          sorted(dic.items(),key=operator.itemgetter(0),reverse=False));
    # 按字典中的键进行降序排序
    print("按键进行降序排序结果为:",\
          sorted(dic.items(),key=operator.itemgetter(0),reverse=True));
    # 按字典中的值进行升序排序
    print("按值进行升序排序结果为:",\
          sorted(dic.items(),key=operator.itemgetter(1),reverse=False));
    # 按字典中的值进行降序排序
    print("按值进行降序排序结果为:",\
          sorted(dic.items(),key=operator.itemgetter(1),reverse=True));
sortDict()