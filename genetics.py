import random
from PIL import Image, ImageDraw
import numpy as np

OFFSET = 10

def generate_point(width, height):
    x = random.randint(0, width)
    y = random.randint(0, height)
    return (x, y)

class Triangle:
    def __init__(self, img_width, img_height, target_image, max_offset=20):
        base_point = generate_point(img_width, img_height)
        self.points = [
            base_point,
            (base_point[0] + random.randint(-max_offset, max_offset), 
             base_point[1] + random.randint(-max_offset, max_offset)),
            (base_point[0] + random.randint(-max_offset, max_offset), 
             base_point[1] + random.randint(-max_offset, max_offset))
        ]
        self.points = [
            (min(max(x, 0), img_width), min(max(y, 0), img_height)) for x, y in self.points
        ]
        x, y = random.randint(0, img_width - 1), random.randint(0, img_height - 1)
        color = target_image.getpixel((x, y))
        self.color = (color[0], color[1], color[2], 128)

class Chromosome:
    def __init__(self, img_height, img_width, target_image, num_triangles):
        self.img_height = img_height
        self.img_width = img_width
        self.background_color = (0, 0, 0, 255)
        self.triangles = [Triangle(img_width, img_height, target_image) for _ in range(num_triangles)]
        self.target_image = target_image

    def mutate(self, mutation_rate=0.05):
        if random.random() < mutation_rate:
            triangle = random.choice(self.triangles)
            if random.choice([True, False]):
                # Slight adjustment to color
                triangle.color = tuple(
                    min(max(c + random.randint(-5, 5), 0), 255) for c in triangle.color
                )
            else:
                # Slight adjustment to position for refinement
                for i in range(2):
                    x, y = triangle.points[i]
                    triangle.points[i] = (
                        min(max(x + random.randint(-5, 5), 0), self.img_width),
                        min(max(y + random.randint(-5, 5), 0), self.img_height),
                    )

    def fine_tune(self):
        # Further fine-tune each triangle's points and colors for lower MSE
        for triangle in self.triangles:
            # Ensure x, y are within bounds before accessing target image pixel
            x, y = triangle.points[0]
            x = min(max(x, 0), self.img_width - 1)
            y = min(max(y, 0), self.img_height - 1)
            target_color = self.target_image.getpixel((x, y))
            triangle.color = tuple(
                min(max(triangle.color[i] + (target_color[i] - triangle.color[i]) // 10, 0), 255)
                for i in range(3)
            )


    def draw(self) -> Image:
        size = self.target_image.size
        img = Image.new('RGB', size, self.background_color)
        draw = Image.new('RGBA', size)
        pdraw = ImageDraw.Draw(draw)
        for triangle in self.triangles:
            pdraw.polygon(triangle.points, fill=triangle.color, outline=triangle.color)
            img.paste(draw, mask=draw)
        return img

    def fitness(self) -> float:
        created_image = np.array(self.draw())
        target_image_array = np.array(self.target_image)
        mse = np.mean((created_image/255 - target_image_array/255) ** 2)
        return -mse

class GeneticAlgorithm:
    def __init__(self, max_width, max_height, target_image, population_size, num_triangles):
        self.population_size = population_size
        self.max_width = max_width
        self.max_height = max_height
        self.population = [Chromosome(max_height, max_width, target_image, num_triangles) for _ in range(population_size)]
        self.target_image = target_image
        self.best_fitness = 0
        self.generations_since_improvement = 0

    def calc_fitnesses(self):
        return [chromosome.fitness() for chromosome in self.population]

    def sort_population(self, fitnesses):
        return [x for _, x in sorted(zip(fitnesses, self.population), key=lambda pair: pair[0], reverse=True)]

    def cross_over(self, parent1, parent2):
        child = Chromosome(self.max_height, self.max_width, self.target_image, len(parent1.triangles))
        child.triangles = [
            tri1 if random.random() < 0.5 else tri2 for tri1, tri2 in zip(parent1.triangles, parent2.triangles)
        ]
        return child

    def run(self, n_generations):
        triangle_count = 50

        for iteration in range(n_generations):
            if self.generations_since_improvement >= 50 and triangle_count < 200:
                triangle_count += 10
                for chromosome in self.population:
                    chromosome.triangles.extend(
                        Triangle(self.max_width, self.max_height, self.target_image) 
                        for _ in range(10)
                    )

            fitnesses = self.calc_fitnesses()
            max_fitness = np.max(fitnesses)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.generations_since_improvement = 0
            else:
                self.generations_since_improvement += 1

            if iteration % 10 == 0:
                print(f"Fitness in Generation {iteration}: mean: {np.mean(fitnesses)}, max: {max_fitness}")

            if iteration % 100 == 0:
                best_chromosome = self.get_best_of_population()
                if best_chromosome:
                    best_image = best_chromosome.draw()
                    best_image.show()
                    print(f"Displaying best image for generation {iteration}")

            sorted_population = self.sort_population(fitnesses)
            elitism_count = 10
            new_population = sorted_population[:elitism_count]
            selected_population = sorted_population[:len(self.population) // 2]

            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected_population, 2)
                child = self.cross_over(parent1, parent2)
                new_population.append(child)

            mutation_rate = 0.05 if self.generations_since_improvement >= 30 else 0.01
            for chromosome in new_population[elitism_count:]:
                chromosome.mutate(mutation_rate=mutation_rate)
                chromosome.fine_tune()  # Apply fine-tuning for additional refinement

            self.population = new_population

    def get_best_of_population(self):
        fitnesses = self.calc_fitnesses()
        sorted_population = self.sort_population(fitnesses)
        return sorted_population[0]

def resize(image, max_size):
    scale_factor = max_size / max(image.size)
    new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
    return image.resize(new_size, Image.Resampling.LANCZOS)

target_image_path = r"C:\Users\NanoCamp\Downloads\target_images\target_images\moon.jpg"
image = Image.open(target_image_path)
image = resize(image, 100)

width, height = image.size
population_size = 100
triangles_number = 200
alg = GeneticAlgorithm(width, height, image, population_size, triangles_number)
alg.run(110)
