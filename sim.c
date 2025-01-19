#include "simulation.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>


#define PI 3.14159265358979323846

#define NUM_THREADS 8

#define DIM 3
#define LENGTH 15.0
#define MASS 1
#define SIGMA 1.0
#define EPSILON 1.0

#define RC 2.5 //(2.5 * SIGMA)
#define DELTA (0.01 * SIGMA)

#define PARTICLES 100
#define TEMPERATURE 2.0
#define VELOCITY_STD sqrt(TEMPERATURE / MASS)

#define TEMP_CHANGES 1000//7
#define TEMP_CHANGE_PROP 0.995//0.999

#define PRE_RUNS 0
#define PRE_RUNS_INTERACTION 0 // number of time steps before data starts being recorded
#define TIME_SCALE 1
#define DP 5
#define DATA_POINTS (DP * TEMP_CHANGES)

#define QUALITY_MULTIPLIER 1
#define RECORDING_STEP (1 * QUALITY_MULTIPLIER) // number of time steps between each data recording
#define DT (1.0 / (60 * QUALITY_MULTIPLIER))

void get_random(double* random_numbers, int n, double low, double high) {
    for (int i = 0; i < n; i++)
        random_numbers[i] = ((double)rand() / RAND_MAX) * (high - low) + low;
}

void sample_normal(double* samples, int n, double mean, double std) {
    double* samples_2 = malloc(n * sizeof(double));
    get_random(samples, n, 0, 1);
    get_random(samples_2, n, 0, 1);
    for (int i = 0; i < n; i++)
        samples[i] = sqrt(-2*log(samples[i])) * cos(2*PI*samples_2[i]) * std + mean;
    free(samples_2);
}

// boundary condition enforcement
double displace(double current_position, int index, double displacement) {
    double new_position = current_position + displacement;
    new_position = fmod(new_position, LENGTH);
    if (new_position < 0)
        new_position += LENGTH;
    return new_position;
}




particle_list* generate_particles(int num_particles) {
    int dofs = PARTICLES * DIM;
    int particles_per_dim = ceil(pow(PARTICLES, 1.0/DIM));
    double ds = LENGTH / particles_per_dim * 0.99;

    double* random_speeds = malloc(dofs * sizeof(double));
    sample_normal(random_speeds, dofs, 0, VELOCITY_STD);
    int p2 = 0; // pointer for random_speeds
    double cm_velocity[DIM] = {0};

    particle_list *pl = malloc(sizeof(particle_list));
    pl->particles = malloc(PARTICLES * sizeof(particle*));
    for (int i = 0; i < PARTICLES; i++) {
        particle *new_particle = malloc(sizeof(particle));
        double* position = malloc(DIM * sizeof(double));
        double* velocity = malloc(DIM * sizeof(double));
        for (int j = 0; j < DIM; j++) { 
            position[j] = ((i / (int)pow(particles_per_dim, j)) % particles_per_dim) * ds; // TODO probably not good
            velocity[j] = random_speeds[p2++];  
            cm_velocity[j] += velocity[j] / PARTICLES;
        }
        new_particle->position = position;
        new_particle->velocity = velocity;
        pl->particles[i] = new_particle;
    }

    // subtract center of mass velocity
    for (int i = 0; i < PARTICLES; i++) {
        for (int j = 0; j < DIM; j++) {
            pl->particles[i]->velocity[j] -= cm_velocity[j];
        }
    }

    pl->num_particles = PARTICLES;

    free(random_speeds);

    return pl;
}

// shifted and truncated L-J force
double get_force(double *force_vector, particle *a, particle *b, double *radius) {
    double potential_energy = 0.0;

    double r_vector[DIM];  
    double r_squared = 0.0;
    for (int i = 0; i < DIM; i++) {
        r_vector[i] = b->position[i] - a->position[i];
        if (fabs(r_vector[i]) > LENGTH / 2) {
            if (r_vector[i] > 0)
                r_vector[i] -= LENGTH;
            else
                r_vector[i] += LENGTH;
        }
        r_squared += r_vector[i] * r_vector[i];
    }

    if (r_squared == 0) {
        printf("program failure\n");
        exit(EXIT_FAILURE);
    }

    double r = sqrt(r_squared);
    *radius = r;
    if (r < RC) { 
        double shifted_r = r + DELTA;
        double sig_r = SIGMA / shifted_r;
        double sig_r6 = pow(sig_r, 6);
        double sig_r12 = sig_r6 * sig_r6;
        double VLJ = 4 * EPSILON * (sig_r12 - sig_r6);
        double VLJ_truncated = VLJ - 4 * EPSILON * (pow(SIGMA / RC, 12) - pow(SIGMA / RC, 6));

        potential_energy = VLJ_truncated;

        double force_magnitude = 24 * EPSILON / SIGMA * (2 * sig_r12 - sig_r6) / shifted_r;
        for (int i = 0; i < DIM; i++) {
            force_vector[i] = force_magnitude * r_vector[i] / r;
        }
    } else {
        for (int i = 0; i < DIM; i++) {
            force_vector[i] = 0; // TODO maybe just return null?
        }
    }
    return potential_energy;
}

void get_forces(particle_list *pl, double ***force_matrix) {

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < PARTICLES-1; i++)
        for (int j = i+1; j < PARTICLES; j++) {  
            double force_vector[DIM] = {0};
            double empty;
            get_force(force_vector, pl->particles[i], pl->particles[j], &empty);
            for (int k = 0; k < DIM; k++) {                    
                force_matrix[i][j][k] = -force_vector[k];
                force_matrix[j][i][k] = force_vector[k];
            }
        }
}

double*** force_matrix;
void forward_system(particle_list *pl, double dt) {
    double hsv[PARTICLES][DIM] = {0.0}; // half-step velocities

    // update the positions
    get_forces(pl, force_matrix);
    
    //#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < PARTICLES; i++) 
        for (int j = 0; j < DIM; j++) {
            double a, velocity, position, new_hsv, new_position;
            a = 0.0;
            //#pragma omp critical
            {
                velocity = pl->particles[i]->velocity[j];
                position = pl->particles[i]->position[j];
            }
            for (int k = 0; k < PARTICLES; k++)
                //# pragma omp atomic
                a += force_matrix[i][k][j];
            a /= MASS;
            new_hsv = velocity + 0.5 * a * dt;
            //# pragma omp critical
            new_position = displace(position, j, dt * new_hsv);
            //#pragma omp critical
            {
                hsv[i][j] = new_hsv;
                pl->particles[i]->position[j] = new_position;
            }
        }
    

    // update the velocities
    get_forces(pl, force_matrix);
    //#pragma omp parallel num_threads(NUM_THREADS)
    for (int i = 0; i < PARTICLES; i++) {
        for (int j = 0; j < DIM; j++) {
            double a = 0.0;
            for (int k = 0; k < PARTICLES; k++)
                a += force_matrix[i][k][j];
            a /= MASS;
            pl->particles[i]->velocity[j] = hsv[i][j] + ((double)1/(double)2)*a*dt;
        }
    }

    // subtract center of mass velocity
    double cm_velocity[DIM] = {0};
    for (int i = 0; i < PARTICLES; i++)
        for (int j = 0; j < DIM; j++)
            cm_velocity[j] += pl->particles[i]->velocity[j] / PARTICLES;
    for (int i = 0; i < PARTICLES; i++) 
        for (int j = 0; j < DIM; j++) 
            pl->particles[i]->velocity[j] -= cm_velocity[j];
    
}

FILE *energy;

double kinetic_energy(particle_list *pl) {
    double total_ke = 0.0;
    for (int i = 0; i < pl->num_particles; i++) {
        double vx = pl->particles[i]->velocity[0];
        double vy = pl->particles[i]->velocity[1];
        double particle_ke = 0.5 * MASS * (vx * vx + vy * vy);
        total_ke += particle_ke;
    }
    return total_ke;
}

void potential_energy_pressure(particle_list *pl, double *data) {
    data[0] = 0.0;
    data[1] = TEMPERATURE * PARTICLES / (pow(LENGTH, DIM));
    double t2 = 0.0;
    double ***force_matrix; 
    for (int i = 0; i < PARTICLES-1; i++)
        for (int j = i+1; j < PARTICLES; j++) {
            double force_vector[DIM] = {0};
            double radius = 0.0;
            data[0] += get_force(force_vector, pl->particles[i], pl->particles[j], &radius);
            double force = 0.0;
            for (int k = 0; k < DIM; k++) {
                force += pow(force_vector[k], 2);
            }
            t2 += sqrt(force)*radius;
        }
    data[1] += t2 / (pow(LENGTH, DIM) * DIM);
    printf("%f\n", t2 / (pow(LENGTH, DIM) * DIM * (PARTICLES * (PARTICLES-1)/2)));
}

void heat_up(particle_list *pl, double prop) {
    printf("Cheese %f\n", prop);
    for (int i = 0; i < PARTICLES; i++)
        for (int j = 0; j < DIM; j++) 
            pl->particles[i]->velocity[j] *= prop;
}

FILE *fp;
void print_pl(particle_list *pl) {
    for (int i = 0; i < PARTICLES; i++) {
        fprintf(fp, "%f", pl->particles[i]->position[0]);
        for (int j = 1; j < DIM; j++) 
            fprintf(fp, " %f", pl->particles[i]->position[j]);
        fprintf(fp, "\n");
    }
}

clock_t start, end;
double cpu_time_used;
struct timespec start_time, end_time;
int main(int argc, const char * argv[]) {
    printf("%f\n", VELOCITY_STD);
    srand(time(NULL));

    fp = fopen("output.txt", "w");
    fprintf(fp, "%d %f %d %d %f %d\n", PARTICLES, LENGTH, DATA_POINTS+1, DIM, TEMPERATURE, TEMP_CHANGES);

    energy = fopen("energy.txt", "w");

    particle_list *pl = generate_particles(PARTICLES);
    force_matrix = malloc(PARTICLES * sizeof(double**));
    for (int i = 0; i < PARTICLES; i++) {
        force_matrix[i] = malloc(PARTICLES * sizeof(double*));
        for (int j = 0; j < PARTICLES; j++) {
            force_matrix[i][j] = malloc(DIM * sizeof(double));
            for (int k = 0; k < DIM; k++)
                force_matrix[i][j][k] = 0.0;
        }
    }    

    clock_gettime(CLOCK_MONOTONIC, &start_time);



    for (int _ = 0; _ < PRE_RUNS_INTERACTION; _++) 
        forward_system(pl, DT/2);

    double data[2];
    for (int i = 0; i < TEMP_CHANGES; i++) {
        // equilibriate
        for (int j = 0; j < PRE_RUNS; j++) {
            forward_system(pl, DT);
        }
        // record
        for (int i = 0; i < DP; i++) {
            print_pl(pl);
            potential_energy_pressure(pl, data);
            fprintf(energy, "%f %f %f\n", kinetic_energy(pl), data[0], data[1]);
            for (int j = 0; j < RECORDING_STEP; j++)
                forward_system(pl, DT);
        }
        printf("%f%%\n", (double) 100*(i+1) / TEMP_CHANGES);
        // change temperature
        heat_up(pl, TEMP_CHANGE_PROP);
    }
    print_pl(pl);

    /*
    for (int k = 0; k < TEMP_CHANGES; k++) {
        for (int i = 0; i < DATA_POINTS; i++) {
            if (i % 15 == 0)
                printf("%f%%\n", (double)i/DATA_POINTS * 100);
            print_pl(pl);
            potential_energy_pressure(pl, data);
            fprintf(energy, "%f %f %f\n", kinetic_energy(pl), data[0], data[1]);
            for (int j = 0; j < RECORDING_STEP; j++)
                forward_system(pl, DT);
        }
    }
    print_pl(pl);
    */

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    printf("Elapsed time: %f seconds\n", elapsed_time);

    fclose(fp);

    free_particle_list(pl);

    for (int i = 0; i < PARTICLES; i++) { 
        for (int j = 0; j < PARTICLES; j++)
            free(force_matrix[i][j]);
        free(force_matrix[i]);
    }
    free(force_matrix);

    return 0;
}

void free_particle_list(particle_list* pl) {
    for (int i = 0; i < PARTICLES; i++) {
        free(pl->particles[i]->position);
        free(pl->particles[i]->velocity);
        free(pl->particles[i]);
    }
    free(pl->particles);
    free(pl);
}
