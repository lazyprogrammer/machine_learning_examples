import numpy as np
import matplotlib.pyplot as plt

# Objective function to minimize (you can change this)
def f(x, y):
    # return np.sin(x) + np.cos(y)
    return -((x - 1)**2 + y**2)

# Evolution Strategies optimizer (simple version)
def evolution_strategies(
    f, bounds, pop_size=50, sigma=0.3, alpha=0.03, iterations=100
):
    dim = 2
    mu = np.random.uniform(bounds[0], bounds[1], size=dim)
    
    history = []

    for gen in range(iterations):
        # Sample noise
        noise = np.random.randn(pop_size, dim)
        population = mu + sigma * noise
        fitness = np.array([f(x[0], x[1]) for x in population])

        history.append((population.copy(), mu.copy()))

        # Normalize fitness for weighting
        fitness_norm = (fitness - np.mean(fitness)) / (np.std(fitness) + 1e-8)
        mu += alpha / (pop_size * sigma) * np.dot(noise.T, fitness_norm)
    
    return history

# Visualization function
def visualize_es(history, bounds, f, resolution=100):
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    plt.figure(figsize=(8, 6))
    for i, (pop, mu) in enumerate(history):
        plt.clf()
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label="f(x, y)")
        plt.scatter(pop[:, 0], pop[:, 1], c='white', s=20, label='Population')
        plt.scatter(mu[0], mu[1], c='red', s=80, label='Mean', edgecolors='black')
        plt.title(f"Evolution Strategies - Step {i+1}")
        plt.xlim(bounds[0], bounds[1])
        plt.ylim(bounds[0], bounds[1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        # plt.pause(0.1)
        plt.waitforbuttonpress()
    plt.show()

# Run
bounds = (-5, 5)
history = evolution_strategies(f, bounds, iterations=80)
visualize_es(history, bounds, f)
