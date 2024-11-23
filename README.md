# Genetic Algorithm for Image Approximation

## Project Overview

This project implements a **genetic algorithm** to approximate a target image using a set of colored triangles. The algorithm progressively evolves a population of solutions by optimizing the arrangement and properties of triangles to closely resemble the target image over successive generations.

---

## Features

### 1. **Core Algorithm**
- Utilizes a genetic algorithm with **mutation**, **crossover**, and **selection** strategies.
- Includes enhancements like **elitism**, **adaptive mutation rates**, and **dynamic triangle addition** to improve convergence efficiency.

### 2. **Fitness Function**
- Measures the similarity between a generated image and the target image using **Mean Squared Error (MSE)**.
- Optimizes fitness by minimizing the error.

### 3. **Triangle-Based Image Representation**
- Each individual solution (chromosome) is composed of multiple triangles.
- Triangles are represented by:
  - Three points (x, y coordinates).
  - RGB color with alpha transparency.

### 4. **Dynamic Adjustments**
- Dynamically increases the number of triangles to improve detail as the algorithm progresses.
- Implements **fine-tuning** to enhance triangle properties for lower MSE.

---

## Key Implementation Details

### Genetic Operators
1. **Crossover**:
   - Combines traits from two parent solutions to create a child.
   - Randomly selects triangles from each parent with a 50% chance.

2. **Mutation**:
   - Introduces random changes to maintain genetic diversity.
   - Modifies triangle colors or positions with a small probability.

3. **Selection**:
   - Employs **elitism** to retain the best solutions across generations.
   - Uses a combination of roulette wheel and tournament selection for creating the next generation.

### Dynamic Triangle Addition
- Gradually increases the number of triangles to enhance image detail as the algorithm converges.

### Fitness Evaluation
- Calculates fitness as the negative MSE between the generated and target images.
- Smaller MSE indicates a closer match to the target image.

---

## Insights from Experiments

### Population Size
- Larger populations lead to better fitness improvement rates by exploring a broader solution space.

| Population Size | Initial Fitness | Fitness (Generation 50) | Fitness (Generation 100) |
|------------------|----------------|--------------------------|---------------------------|
| 50               | -0.199         | -0.068                  | -0.025                   |
| 100              | -0.203         | -0.064                  | -0.020                   |
| 150              | -0.202         | -0.081                  | -0.022                   |

### Triangle Count
- Increasing the number of triangles allows for finer details and better approximation of the target image.

---

## Future Enhancements
1. **Parallelization**: Speed up fitness evaluation by leveraging parallel computing.
2. **Interactive Visualization**: Include a real-time visualization of the algorithm's progress.
3. **Support for Different Shapes**: Extend the algorithm to support polygons other than triangles for image reconstruction.

---

Feel free to experiment with the parameters and share your results! 🎨
