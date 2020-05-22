import numpy as np
import matplotlib.pyplot as plt
import random

class Particle(object):
    def __init__(self, v, x):
        self.v = v                    # 粒子当前速度
        self.x = x                    # 粒子当前位置
        self.pbest = x                # 粒子历史最优位置
        
class PSO(object):
    def __init__(self, interval, tab='min', Num=10, iterMax=1000, w=1, c1=1, c2=1):
        self.interval = interval                                            # 给定求解空间
        self.tab = tab.strip()                                              # 求解最大值还是最小值的标签: 'min' - 最小值；'max' - 最大值
        self.iterMax = iterMax                                              # 迭代求解次数
        self.w = w                                                          # 惯性因子
        self.c1, self.c2 = c1, c2                                           # 学习因子
        self.v_max = (interval[1] - interval[0]) * 0.1                      # 设置最大速度
        #####################################################################
        self.particle_list, self.gbest = self.initPartis(Num)               # 完成粒子群的初始化，并提取群体历史最优位置
        self.x_seeds = np.array(list(particle.x for particle in self.particle_list))           # 提取粒子群的种子状态
        self.solve()                                                        # 完成主体的求解过程
        self.display()                                                      # 数据可视化展示
        
    def initPartis(self, Num):
        particle_list = list()
        for i in range(Num):
            v_seed = random.uniform(-self.v_max, self.v_max)
            x_seed = random.uniform(*self.interval)
            particle_list.append(Particle(v_seed, x_seed))
        temp = 'find_' + self.tab
        if hasattr(self, temp):                                             # 采用反射方法提取对应的函数
            gbest = getattr(self, temp)(particle_list)
        else:
            exit('>>>tab标签传参有误："min"|"max"<<<')
        return particle_list, gbest
        
    def solve(self):
        for i in range(self.iterMax):
            for particle in self.particle_list:
                f1 = self.func(particle.x)
                # 更新粒子速度，并限制在最大速度之内
                particle.v = self.w * particle.v + self.c1 * random.random() * (particle.pbest - particle.x) + self.c2 * random.random() * (self.gbest - particle.x)
                if particle.v > self.v_max:
                    particle.v = self.v_max
                elif particle.v < -self.v_max:
                    particle.v = -self.v_max
                # 更新粒子位置，并限制在待解区间之内
                if self.interval[0] <= particle.x + particle.v <=self.interval[1]:
                    particle.x = particle.x + particle.v 
                else:
                    particle.x = particle.x - particle.v                    # 不让粒子超出限制空间之外有很多方法
                f2 = self.func(particle.x)
                getattr(self, 'deal_'+self.tab)(f1, f2, particle)           # 更新粒子历史最优位置与群体历史最优位置      
        
    def func(self, x):                                                      # 适应度函数
        value = np.sin(x**2) * (x**2 - 2*x)
        return value
        
    def find_min(self, particle_list):                                      # 按适应度函数最小值找到粒子群初始化的历史最优位置
        p = min(particle_list, key=lambda p: self.func(p.pbest))
        return p.pbest
        
    def find_max(self, particle_list):                                      # 按适应度函数最大值找到粒子群初始化的历史最优位置
        p = max(particle_list, key=lambda p: self.func(p.pbest))
        return p.pbest
        
    def deal_min(self, f1, f2, particle):
        if f2 < f1:                          # 更新粒子历史最优位置
            particle.pbest = particle.x
        if f2 < self.func(self.gbest):
            self.gbest = particle.x          # 更新群体历史最优位置
            
    def deal_max(self, f1, f2, particle):
        if f2 > f1:                          # 更新粒子历史最优位置
            particle.pbest = particle.x
        if f2 > self.func(self.gbest):
            self.gbest = particle.x          # 更新群体历史最优位置
            
    def display(self):
        print('solution: {}'.format(self.gbest))
        plt.figure(figsize=(8, 4))
        x = np.linspace(self.interval[0], self.interval[1], 300)
        y = self.func(x)
        plt.plot(x, y, 'g-', label='function')
        plt.plot(self.x_seeds, self.func(self.x_seeds), 'b.', label='seeds')
        plt.plot(self.gbest, self.func(self.gbest), 'r*', label='solution')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('solution = {}'.format(self.gbest))
        plt.legend()
        plt.savefig('PSO.png', dpi=500)
        plt.show()
        plt.close()
        
if __name__ == '__main__':
    PSO([-10, 10], 'max')