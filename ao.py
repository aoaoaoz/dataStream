import numpy


def select():
    delList = [2, 3, 4, 7, 12, 14, 15, 21, 22, 42]
    delList.reverse()

    result = []

    with open('I:\数据流聚类算法实现\kddcup.data_10_percent_corrected', 'r') as f:
        for i in range(400000):
            line = f.readline()
            line = line.split(',')
            for j in delList:
                del line[j-1]
            print(i)
            line = ','.join(line)
            result.append(line)

    with open('aaa.csv', 'w') as f:
        f.write( '\n'.join(result) )

def norm(path):
    result = []
    with open(path, 'r') as f:
        lines = f.readlines()
        size = len(lines[0].strip().split(','))
        Max = size * [0]
        cnt = 0
        for line in lines:
            print(cnt)
            cnt += 1
            line = line.strip()
            line = line.split(',')
            for i in range(size):
                Max[i] = max(Max[i], float(line[i]))
        for line in lines:
            print(cnt)
            cnt -= 1
            line = line.strip()
            line = line.split(',')
            for i in range(size):
                if float(Max[i]) > 0.01:
                    line[i] = str(float(line[i])*1.0/Max[i])
            result.append(','.join(line))
        print(Max)
    with open('bbb.csv', 'w') as wf:
        wf.write('\n'.join(result))



norm('aaa.csv')