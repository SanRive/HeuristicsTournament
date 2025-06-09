def run_de(strategy, func_name, F, CR, seed=0, D=10, NP=30, time_limit=10):
    np.random.seed(seed)

    def rastrigin(x): A = 10; return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)
    def ackley(x):
        a, b, c = 20, 0.2, 2*np.pi
        d = len(x)
        return -a*np.exp(-b*np.sqrt(np.sum(x**2)/d)) - np.exp(np.sum(np.cos(c*x))/d) + a + np.exp(1)
    def griewank(x):
        return np.sum(x**2)/4000 - np.prod([np.cos(xi/np.sqrt(i+1)) for i, xi in enumerate(x)]) + 1
    func_map = {"rastrigin": rastrigin, "ackley": ackley, "griewank": griewank}
    fobj = func_map[func_name]

    lim_inf, lim_sup = -5.12, 5.12
    poblacion = np.random.uniform(lim_inf, lim_sup, (NP, D))
    fitness = np.array([fobj(ind) for ind in poblacion])
    best = poblacion[np.argmin(fitness)]

    start_time = time.time()
    while time.time() - start_time < time_limit:
        for i in range(NP):
            if time.time() - start_time >= time_limit:
                break
            idxs = [idx for idx in range(NP) if idx != i]
            a, b, c, d_, e = poblacion[np.random.choice(idxs, 5, replace=False)]
            x = poblacion[i]
            if strategy == "rand/1":
                mutant = a + F * (b - c)
            elif strategy == "best/1":
                mutant = best + F * (a - b)
            elif strategy == "current-to-best/1":
                mutant = x + F * (best - x) + F * (a - b)
            elif strategy == "rand/2":
                mutant = a + F * (b - c) + F * (d_ - e)
            else:
                raise ValueError("Unknown strategy")
            j_rand = np.random.randint(D)
            trial = np.array([mutant[j] if np.random.rand() < CR or j == j_rand else x[j] for j in range(D)])
            trial = np.clip(trial, lim_inf, lim_sup)
            f_trial = fobj(trial)
            if f_trial < fitness[i]:
                poblacion[i] = trial
                fitness[i] = f_trial
                if f_trial < fobj(best): best = trial
    best_fitness = np.min(fitness)
    return best_fitness
