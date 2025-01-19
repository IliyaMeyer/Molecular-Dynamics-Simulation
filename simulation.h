#ifndef SIMULATION_H
#define SIMULATION_H

// populate <random_numbers> with <n> random numbers between <low> and <high>
void get_random(double* random_numbers, int n, double low, double high);

// populate <samples> with <n> numbers sampled from a normal distribution with <mean> and <std>
void sample_normal(double* samples, int n, double mean, double std);

// individual particle object
typedef struct {
    double* position;
    double* velocity;
} particle;

// wrapper for particles
typedef struct {
    particle** particles;
    int num_particles;
} particle_list;

// return list with <num_particles> of particles
particle_list* generate_particles(int num_particles);

// forward the positions of <current_state> by <dt> based on the respective velocities
void forward_position(particle_list* state, double dt);

// forward the velocities of <current_state> by <dt> based on the relevant force regime
void forward_velocity(particle_list* state, double dt);

// free particle_list struct
void free_particle_list(particle_list* particle_list);

#endif // SIMULATION_H
