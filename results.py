

Results = {'Rand':
               {'Exp1seed': [0.228, 0.556, 0.666, 0.646, 0.618, 0.684, 0.702, 0.704, 0.732, 0.704, 0.73, 0.768],
                'Exp2seed': [0.228, 0.546, 0.444, 0.456, 0.568, 0.642, 0.458, 0.682, 0.71, 0.694, 0.722, 0.764],
                'Exp3seed': [0.228, 0.596, 0.484, 0.512, 0.628, 0.752, 0.728, 0.75, 0.756, 0.764, 0.772, 0.768],
                'Exp1noseed': [0.228, 0.382, 0.58, 0.634, 0.684, 0.646, 0.708, 0.69, 0.698, 0.702, 0.624] ,
                'Exp2noseed': [0.228, 0.302, 0.422, 0.506, 0.604, 0.702, 0.708, 0.678, 0.528, 0.514, 0.714],
                'Exp3noseed': [0.228, 0.566, 0.506, 0.616, 0.67, 0.656, 0.666, 0.7, 0.694, 0.68, 0.68]  },
            'ConfScor':
                {'Exp1seed': [0.228, 0.428, 0.494, 0.418, 0.406, 0.432, 0.468, 0.708, 0.616, 0.68, 0.702, 0.478],
                'Exp2seed': [0.228, 0.608, 0.7, 0.386, 0.5, 0.436, 0.498, 0.622, 0.462, 0.494, 0.452, 0.47],
                'Exp3seed': [0.228, 0.492, 0.42, 0.632, 0.618, 0.634, 0.576, 0.636, 0.748, 0.63, 0.74, 0.754],
                'Exp1noseed': [0.228, 0.252, 0.444, 0.48, 0.536, 0.638, 0.638, 0.53, 0.656, 0.494, 0.694],
                'Exp2noseed': [0.228, 0.256, 0.238, 0.392, 0.396, 0.598, 0.588, 0.55, 0.612, 0.69, 0.642],
                'Exp3noseed':  [0.228, 0.278, 0.53, 0.604, 0.656, 0.584, 0.678, 0.67, 0.644, 0.668, 0.686] },
            'ExpGrad':
                {'Exp1seed': [0.228, 0.654, 0.64, 0.338, 0.344, 0.34, 0.424, 0.518, 0.462, 0.69, 0.614, 0.598],
                'Exp2seed': [0.228, 0.652, 0.458, 0.388, 0.414, 0.45, 0.58, 0.748, 0.718, 0.728, 0.68, 0.734],
                'Exp3seed': [0.228, 0.42, 0.384, 0.598, 0.614, 0.662, 0.668, 0.616, 0.702, 0.694, 0.724, 0.71],
                'Exp1noseed': [0.228, 0.632, 0.622, 0.422, 0.422, 0.414, 0.58, 0.6, 0.532, 0.632, 0.442],
                'Exp2noseed': [0.228, 0.296, 0.398, 0.432, 0.478, 0.632, 0.726, 0.72, 0.7, 0.728, 0.712],
                'Exp3noseed': [0.228, 0.286, 0.322, 0.34, 0.332, 0.38, 0.362, 0.344, 0.438, 0.438, 0.464]   },
            'CoreSet':
                {'Exp1seed': [0.228, 0.442, 0.388, 0.456, 0.666, 0.516, 0.512, 0.5, 0.518, 0.518, 0.562, 0.562],
                'Exp2seed': [0.228, 0.532, 0.52, 0.434, 0.414, 0.454, 0.46, 0.446, 0.46, 0.496, 0.49, 0.484],
                'Exp3seed': [0.228, 0.598, 0.494, 0.486, 0.464, 0.49, 0.5, 0.482, 0.514, 0.49, 0.51, 0.52],
                'Exp1noseed': [0.228, 0.276, 0.366, 0.424, 0.446, 0.452, 0.456, 0.474, 0.586, 0.482, 0.49],
                'Exp2noseed': [0.228, 0.31, 0.334, 0.412, 0.616, 0.626, 0.618, 0.56, 0.5, 0.492, 0.644],
                'Exp3noseed': [0.228, 0.33, 0.37, 0.422, 0.448, 0.454, 0.44, 0.47, 0.468, 0.482, 0.488]  } }

Averages = {}
for Alg in ['Rand', 'ConfScor', 'ExpGrad', 'CoreSet']:
    isitseed = {}
    for isSeed in ['seed', 'noseed']:
        list = []
        for i, el in enumerate(Results[Alg][f'Exp1{isSeed}']):
            list.append((Results[Alg][f'Exp1{isSeed}'][i]+Results[Alg][f'Exp2{isSeed}'][i]+Results[Alg][f'Exp2{isSeed}'][i])/3)
        isitseed[isSeed] = list
    Averages[Alg] = isitseed

print(Averages)



NoSeedActiveAvg = []
SeedActiveAvg = []
for i in range(11):
    NoSeedActiveAvg.append((Averages['ConfScor']['noseed'][i]+Averages['CoreSet']['noseed'][i]+Averages['ExpGrad']['noseed'][i])/3)
for i in range(12):
    SeedActiveAvg.append((Averages['ConfScor']['seed'][i]+Averages['CoreSet']['seed'][i]+Averages['ExpGrad']['seed'][i])/3)

from matplotlib import pyplot as plt
plt.plot(range(11), Averages['Rand']['noseed'], label = 'Random without seed')
plt.plot(range(11), Results['Rand']['Exp1noseed'], label = 'First training run Random without seed')
plt.plot(range(11), Results['Rand']['Exp2noseed'], label = 'Second training run Random without seed')
plt.plot(range(11), Results['Rand']['Exp3noseed'], label = 'Third training run Random without seed')
plt.legend()
plt.show()


