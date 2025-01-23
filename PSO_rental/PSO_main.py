import random
import numpy as np
from PSO_rental.Inventory_simulator_rental import *
from Utils.ScenarioDesign_rental import ScenarioDesign_rental

def evaluate_function(seed,parameters,individual):
    """
    Evaluate the fitness of a particle based on its position.
    The position is a vector of the form [[q1, [i1, j1, ...]], ..., [qT, [iT, jT, ...]]].
    """
    # get the inventory environment
    env = InvOptEnv(seed,parameters)
    fitness = env.PSO_run(individual)
    return fitness

def initialize_particle(T, num_retailer, m, n, index_range):
    """
    Initialize a particle with random values.
    """
    # num_index = random.randint(1, 3)
    # num_index = 2
    decisions = []
    num_rental_choice = index_range[-1]
    rental_choice = [i for i in range(num_rental_choice)]
    for _ in range(T):
        decision = []
        replenishment_decisions = []
        for _ in range(num_retailer):
            replenishment_decisions.append(round(np.random.uniform(m, n)))
        decision.append(replenishment_decisions)
        perm = np.random.permutation(rental_choice)
        num_rental = np.random.randint(num_rental_choice)
        rental_decisions = perm[:num_rental].tolist()
        while len(rental_decisions) < num_rental_choice:
            rental_decisions.append(0) # 0 means the rental decision of []
        decision.append(rental_decisions)
        decisions.append(decision)
    return decisions
    # original with fixed number of rental decisions
    # return [
    #     [[random.uniform(m, n) for _ in range(num_retailer)], [random.randint(*index_range) for _ in range(num_index)]]
    #     for _ in range(T)
    # ]

def update_velocity(velocity, position, personal_best, global_best, w, c1, c2):
    """
    Update the velocity of a particle.
    """
    new_velocity = []
    for v, pos, p_best, g_best in zip(velocity, position, personal_best, global_best):
        quantities_v, indices_v = v
        quantities_pos, indices_pos = pos
        quantities_p_best, indices_p_best = p_best
        quantities_g_best, indices_g_best = g_best

        # Update velocities for quantities
        quantities_v = [
            round(w * quantities_v[i] + c1 * random.random() * (quantities_p_best[i] - quantities_pos[i]) + c2 * random.random() * (quantities_g_best[i] - quantities_pos[i]))
            for i in range(len(quantities_pos))
        ]

        # Update velocities for indices
        indices_v = [
            w * indices_v[i] + c1 * random.random() * (indices_p_best[i] - indices_pos[i]) + c2 * random.random() * (indices_g_best[i] - indices_pos[i])
            for i in range(len(indices_pos))
        ]

        new_velocity.append([quantities_v, indices_v])
    return new_velocity

def update_position(position, velocity, m, n, index_range):
    """
    Update the position of a particle based on its velocity.
    """
    new_position = []
    for pos, vel in zip(position, velocity):
        quantities_pos, indices_pos = pos
        quantities_vel, indices_vel = vel

        # Update quantities and clamp to [m, n]
        quantities = [
            min(max(quantities_pos[i] + quantities_vel[i], m), n) for i in range(len(quantities_pos))
        ]

        # Update indices and clamp to index_range
        indices = [
            max(min(int(indices_pos[i] + indices_vel[i]), index_range[1]-1), index_range[0])
            for i in range(len(indices_pos))
        ]

        new_position.append([quantities, indices])
    return new_position

def pso_optimize(randomSeed_ngen, seedRotate, seed, parameters, T, num_retailer, m, n, index_range, num_particles, max_iterations, w, c1, c2):
    """
    Particle Swarm Optimization implementation.
    """
    # Initialize particles and velocities
    particles = [initialize_particle(T, num_retailer, m, n, index_range) for _ in range(num_particles)]
    velocities = []
    for particle in particles:
        velocity = [
            [[0.0, 0.0], [0.0] * len(p[1])] for p in particle
        ]
        velocities.append(velocity)

    instance_seed = randomSeed_ngen[0]
    print("Instance seed: ", instance_seed)

    # Initialize personal and global bests
    personal_bests = particles[:]
    personal_best_scores = [evaluate_function(instance_seed, parameters, p) for p in particles]
    global_best = personal_bests[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)

    for iteration in range(1, max_iterations+1):
        # Added by mengxu to do seed rotation
        if seedRotate:
            instance_seed = randomSeed_ngen[iteration]
        print("Instance seed: ", instance_seed)

        for i, particle in enumerate(particles):
            # Update velocity and position
            velocities[i] = update_velocity(
                velocities[i], particle, personal_bests[i], global_best, w, c1, c2
            )
            particles[i] = update_position(particle, velocities[i], m, n, index_range)

            # Evaluate the particle's fitness
            score = evaluate_function(instance_seed, parameters, particles[i])

            # Update personal best if necessary
            if score < personal_best_scores[i]:
                personal_bests[i] = particles[i]
                personal_best_scores[i] = score

                # Update global best if necessary
                if score < global_best_score:
                    global_best = particles[i]
                    global_best_score = score

        print(f"Iteration {iteration + 1}/{max_iterations}, Best Score: {global_best_score}")

    return global_best, global_best_score



def main(dataset_name, seed):
# if __name__ == "__main__":
#     dataset_name = str(sys.argv[1])
#     seed = int(sys.argv[2])
    num_particles = 400
    max_iterations = 50
    seed_rotation = False

    random.seed(int(seed))
    np.random.seed(int(seed))
    randomSeed_ngen = []
    for i in range((max_iterations + 1)):
    # for i in range((ngen+1)*ins_each_gen): # the *ins_each_gen is added by mengxu followed the advice of Meng 2022.11.01
        randomSeed_ngen.append(np.random.randint(2000000000))

    # get parameters for the given dataset/scenario
    scenarioDesign = ScenarioDesign_rental(dataset_name)
    parameters = scenarioDesign.get_parameter()
    num_retailer = parameters['num_retailer']
    T = parameters['epi_len'] + 1 # Number of time periods
    max_index_range = len(parameters['rental_choice'])
    max_retailer_capacity = max(parameters['capacity'])

    # Example usage
    # T = 5  # Number of time periods
    m, n = 0, max_retailer_capacity  # Range for q for each retailer
    index_range = (0, max_index_range)  # Range for index values
    w = 0.5  # Inertia weight
    c1 = 1.5  # Cognitive component
    c2 = 1.5  # Social component

    best_position, best_score = pso_optimize(randomSeed_ngen, seed_rotation, seed, parameters, T, num_retailer, m, n, index_range, num_particles, max_iterations, w, c1, c2)
    print("Best Position:", best_position)
    print("Best Score:", best_score)