def add(l):
    l.append(2)

j = []
add(j)
add(j)
print(j)

bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
        {'name': 'min_child_weight', 'type': 'continuous', 'domain': (1, 100)},
        {'name': 'colsample_bytree', 'type': 'continuous', 'domain': (0.1, 0.8)},
        {'name': 'subsample', 'type': 'continuous', 'domain': (0.1, 0.8)}  
            ]

a=map(lambda x:x['domain'],bds)
print(list(a))