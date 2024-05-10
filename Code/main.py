import numpy as np
import matplotlib.pyplot as plt  # 绘图
import scipy.optimize as sco
import read as rd

#资产个数
PROPERTY_N = 48    
#选择资产的个数
SELECT_N = 10 

#ExpReturn：组合中每个资产的预期收益率
#ExpCovariance: 组合中证券的协方差矩阵
#PortWts: 资产权重
#PortRisk: 资产组合风险
#PortReturn: 资产组合预期收益

def frontcon(ExpReturn, ExpCovariance, NumPorts):
    noa = len(ExpReturn)

    def statistics(weights):
        weights = np.array(weights)
        z = np.dot(ExpCovariance, weights)
        x = np.dot(weights, z)
        port_returns = (np.sum(ExpReturn * weights.T))
        port_variance = np.sqrt(x)
        num1 = port_returns / port_variance
        return np.array([port_returns, port_variance, num1])

    # 定义一个函数对方差进行最小化
    def min_variance(weights):
        return statistics(weights)[1]

    #在特定的收益率下，求得使variance最小的权重，存入PortWts，并将最小的variance存入target_variance中
    bnds = tuple((0, 1) for x in range(noa))
    #在收益率的区间中等间隔地取NumPorts个值，放入target_returns中
    target_returns = np.linspace(min(ExpReturn), max(ExpReturn), NumPorts)     
    target_variance = []
    PortWts = []
    for tar in target_returns:
        # 在最优化时采用两个约束，1.给定目标收益率，2.投资组合权重和为1。
        # 在不同目标收益率水平（target_returns）循环时，最小化的一个约束条件会变化。
        cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tar}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        res = sco.minimize(min_variance, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)
        target_variance.append(res['fun'])
        PortWts.append(res["x"])
    target_variance = np.array(target_variance)
    return [target_variance, target_returns, PortWts]

#找到给定风险系数下的最大收益率
def search_max(target_variance,min_variance,Riskfactor):
    if(target_variance[min_variance]> Riskfactor):
        return min_variance
    for i in range(min_variance,len(target_variance)):
        if(target_variance[i] <= Riskfactor and target_variance[i+1] > Riskfactor):
            return i

#找出并返回权值最大的SELECT_N个资产的索引并返回新的权值矩阵
def select_index(PortWts,number,ExpReturn):
    index=np.argsort(PortWts[number])[:-SELECT_N-1:-1]
    select_wt=[]
    finalreturns=[]
    for i in range(len(index)):
       select_wt.append(PortWts[number][index[i]])
       finalreturns.append(ExpReturn[index[i]])
    select_wt=np.array(select_wt)
    finalreturns=np.array(finalreturns)
    return [index,select_wt,finalreturns]

#对取出的SELECT_N个权值进行归一化
def normalize(data):
    sum_wt=np.sum(data)
    for i in range(len(data)):
        data[i]=data[i] / sum_wt
    return data

#portstats：收益与风险计算函数    
def portstats(ExpReturn, ExpCovariance, PortWts):
    PortRisk = 0
    for i in range(len(PortWts)):
        for j in range(len(PortWts)):
            PortRisk += PortWts[i] * PortWts[j] * ExpCovariance[i][j]
    PortReturn = np.sum(ExpReturn * np.array(PortWts))
    return PortReturn,np.sqrt(PortRisk)

#得到取出的SELECT_N个资产的协方差矩阵
def deal_Cov(ExpCovariance,index):
    TransCov=[]
    NewCov=[]
    for i in (range(len(index))):
        TransCov.append(ExpCovariance[index[i]])
    TransCov=np.array(TransCov)
    for i in (range(len(index))):
        #np.insert(NewCov,len(index),values=TransCov[:,index[i]],axis=1)
        NewCov.append(TransCov[:,index[i]])
    NewCov=np.array(NewCov)
    return NewCov


ExpReturn = np.array([1.001971429,0.995466667,1.000980952,1.00052381,1.002452381,1.001771429,1.001371429,1.002680952,1.00247619,1.002385714,0.99842381,1.000314286,0.998447619,1.000280952,0.99772381,1.000861905,1.001538095,1.000985714,1.002547619,0.999280952,1.013033333,1.002971429,0.991942857,0.997290476,0.998238095,0.998942857,1.018971429,0.999952381,1.001,1.002419048,0.999104762,0.997019048,1.0012,0.999147619,0.999438095,1.000685714,1.006609524,1.001147619,0.998095238,0.999366667,1.000357143,1.000847619,0.996666667,1.001985714,1.000680952,0.996347619,1.007942857,1.000595238])
ExpCovariance = rd.dat_to_matrix("D:\运筹学\err_cov.txt")
NumPorts = 200

[target_variance, target_returns, PortWts] =frontcon(ExpReturn, ExpCovariance, NumPorts)

#风险最低的方案的索引
min_variance=np.argmin(target_variance,axis=0)

#3种投资类型所对应的点的索引
conservative=search_max(target_variance,min_variance,0.012)
aggressive=search_max(target_variance,min_variance,0.020)
neutrality=search_max(target_variance,min_variance,0.016)

print("****************************")
print("保守型：                  预期收益                  风险")
[c_index,c_wt,c_returns]=select_index(PortWts,conservative,ExpReturn)
NewCov_c=deal_Cov(ExpCovariance,c_index)
[finalreturns_c,finalrisk_c]=portstats(c_returns, NewCov_c,normalize(c_wt))
print("原来:              ",target_returns[conservative],"    ",target_variance[conservative])
print("归一化后:          ",finalreturns_c,"    ",finalrisk_c)
print("所选择的公司的序号：",c_index+1)
print("权重:              ",normalize(c_wt))

print("****************************")
print("激进型：                  预期收益                  风险")
[a_index,a_wt,a_returns]=select_index(PortWts,aggressive,ExpReturn)
NewCov_a=deal_Cov(ExpCovariance,a_index)
[finalreturns_a,finalrisk_a]=portstats(a_returns, NewCov_a, normalize(a_wt))
print("原来:              ",target_returns[aggressive],"     ",target_variance[aggressive])
print("归一化后:          ",finalreturns_a,"    ",finalrisk_a)
print("所选择的公司的序号：",a_index+1)
print("权重:              ",normalize(a_wt))

print("****************************")
print("中立型：                  预期收益                  风险")
[n_index,n_wt,n_returns]=select_index(PortWts,neutrality,ExpReturn)
NewCov_n=deal_Cov(ExpCovariance,n_index)
[finalreturns_n,finalrisk_n]=portstats(n_returns, NewCov_n, normalize(n_wt))
print("原来:              ",target_returns[neutrality],"    ",target_variance[neutrality])
print("归一化后:          ",finalreturns_n,"    ",finalrisk_n)
print("所选择的公司的序号：",n_index+1)
print("权重:              ",normalize(n_wt))


plt.plot(target_variance, target_returns)
plt.title("Mean-Variance-Efficient Frontier")
plt.xlabel("Risk(Standard Deviation)")
plt.ylabel("Expected Return")
plt.show()




