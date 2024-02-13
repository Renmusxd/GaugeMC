__device__ int get_thread_number() {
    int threadsPerBlock  = blockDim.x * blockDim.y;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;
    int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;
    return globalThreadNum;
}

int __device__ calculate_index_up_difference(int bound, int index, int delta)
{
    int up_index_diff = ((index + 1)%bound - index) * delta;
    return up_index_diff;
}

int __device__ calculate_index_down_difference(int bound, int index, int delta)
{
    int down_index_diff = ((index + bound - 1)%bound - index) * delta;
    return down_index_diff;
}


__device__ int get_edge_sum_from_plaquettes(int* plaquette_buffer, int edge_type,
        int coord_index, int* coords, int* bounds, int* deltas)
{
    int other_edges_a = 0;
    int other_edges_b = 0;
    int other_edges_c = 0;
    int plaquettes_types_a = 0;
    int plaquettes_types_b = 0;
    int plaquettes_types_c = 0;
    // Plaquette types:
    // 0: tx (0,1)
    // 1: ty (0,2)
    // 2: tz (0,3)
    // 3: xy (1,2)
    // 4: xz (1,3)
    // 5: yz (2,3)
    switch(edge_type) {
        case 0:
            other_edges_a = 1; other_edges_b = 2; other_edges_c = 3;
            plaquettes_types_a = 0; plaquettes_types_b = 1; plaquettes_types_c = 2;
            break;
        case 1:
            other_edges_a = 0; other_edges_b = 2; other_edges_c = 3;
            plaquettes_types_a = 0; plaquettes_types_b = 3; plaquettes_types_c = 4;
            break;
        case 2:
            other_edges_a = 0; other_edges_b = 1; other_edges_c = 3;
            plaquettes_types_a = 1; plaquettes_types_b = 3; plaquettes_types_c = 5;
            break;
        case 3:
            other_edges_a = 0; other_edges_b = 1; other_edges_c = 2;
            plaquettes_types_a = 2; plaquettes_types_b = 4; plaquettes_types_c = 5;
            break;
        default:
            return ~0;
    }
    int other_edges[] = {other_edges_a, other_edges_b, other_edges_c};
    int plaquette_types[] = {plaquettes_types_a, plaquettes_types_b, plaquettes_types_c};

    int global_index = coord_index * 6;

    int sum = 0;
    for (int i = 0; i < 3; i++) {
        int other_edge = other_edges[i];
        int plaquette_type = plaquette_types[i];

        int plaquette_index = global_index + plaquette_type;
        int down_index = calculate_index_down_difference(bounds[other_edge], coords[other_edge], deltas[other_edge]) + plaquette_index;
        int down_diff = plaquette_buffer[plaquette_index] - plaquette_buffer[down_index];

        int sign = 1;
        if (edge_type > other_edge) {
            sign = -1;
        }

        sum += sign * down_diff;
    }

    return sum;
}


extern "C" __global__ void sum_buffer(float* buffer, int num_threads, int num_steps)
{
    int globalThreadNum = get_thread_number();
    if (globalThreadNum >= num_threads) {
        return;
    }
    for (int i = 1; i < num_steps; i++) {
        buffer[globalThreadNum] += buffer[globalThreadNum + (i * num_threads)];
        buffer[globalThreadNum + (i * num_threads)] = 0.0;
    }
}

extern "C" __global__ void global_update_sweep(int* plaquette_buffer,
        float* potential_buffer, float* chemical_potential_buffer, int* potential_redirect, int potential_vector_size,
        float* rng_buffer, int replicas, int t, int x, int y, int z)
{
    // tx : y * z
    // ty : x * z
    // tz : x * y
    // xy : t * z
    // xz : t * y
    // yz : t * x
    int num_planes_per_replica = t*(x+y+z) + x*(y+z) + y*z;
    int globalThreadNum = get_thread_number();
    if (globalThreadNum >= replicas * num_planes_per_replica) {
        return;
    }

    int replica_index = globalThreadNum / num_planes_per_replica;
    int inside_replica_index = globalThreadNum % num_planes_per_replica;
    int planes_per_type[] = {y*z, x*z, x*y, t*z, t*y, t*x};
    int dec = inside_replica_index;
    int p = 0;
    for (p = 0; p < 6 && dec >= planes_per_type[p]; p++) {
        dec -= planes_per_type[p];
    }
    // We are in plane type p, with local index dec.
    int plane_one_arr[] = {0, 0, 0, 1, 1, 2};
    int plane_two_arr[] = {1, 2, 3, 2, 3, 3};
    int index_one_arr[] = {2, 1, 1, 0, 0, 0};
    int index_two_arr[] = {3, 3, 2, 3, 2, 1};

    int plaquettes_per_txyzslice = 6;
    int plaquettes_per_txyslice = z * plaquettes_per_txyzslice;
    int plaquettes_per_txslice = y * plaquettes_per_txyslice;
    int plaquettes_per_tslice = x * plaquettes_per_txslice;
    int strides[] = {plaquettes_per_tslice, plaquettes_per_txslice, plaquettes_per_txyslice, plaquettes_per_txyzslice};
    int bounds[] = {t,x,y,z};

    // Plane one/two are 0 and 1 if dealing with a tx plane
    int plane_one = plane_one_arr[p];
    int plane_two = plane_two_arr[p];
    // Indexed dim one/two are 2 and 3 if dealing with a tx plane
    int indexed_dim_one = index_one_arr[p];
    int indexed_dim_two = index_two_arr[p];

    // First get the actual indices for indexed dims
    // If dec is between 0 and bounds[a] * bounds[b].
    // Divide by bounds[b] to get a_index, and mod for b_index
    int a_index = dec / bounds[indexed_dim_two];
    int b_index = dec % bounds[indexed_dim_two];

    // Get offsets for replica and for indexing.
    int replica_offset = replica_index * (t*x*y*z*6);
    int potential_index = potential_redirect[replica_index];
    int potential_offset = potential_index * potential_vector_size;

    int plane_index_offset = a_index * strides[indexed_dim_one] + b_index * strides[indexed_dim_two];

    float boltzman_weights[3] = {0.0,0.0,0.0};
    for (int i = 0; i < bounds[plane_one]; i++) {
        for (int j = 0; j < bounds[plane_two]; j++) {
            // Iterate over plane
            int offset = i*strides[plane_one] + j*strides[plane_two];
            int np = plaquette_buffer[replica_offset + plane_index_offset + offset + p];
            for (int k = 0; k < 3; k++) {
                boltzman_weights[k] += potential_buffer[potential_offset + abs(np+k-1)];
            }
        }
    }
    // Add chemical potential
    boltzman_weights[0] += chemical_potential_buffer[potential_index] * planes_per_type[p];
    boltzman_weights[2] -= chemical_potential_buffer[potential_index] * planes_per_type[p];

    float min_potential = min(min(boltzman_weights[0], boltzman_weights[1]), boltzman_weights[2]);
    float total_weight = 0.0;
    for (int i = 0; i < 3; i++) {
        boltzman_weights[i] = exp(-boltzman_weights[i] + min_potential);
        total_weight += boltzman_weights[i];
    }

    float rng = rng_buffer[globalThreadNum] * total_weight;
    int j;
    for (j = 0; j < 3; j++) {
        rng -= boltzman_weights[j];
        if (rng <= 0.0) {
            break;
        }
    }

    int delta = j-1;
    for (int i = 0; i < bounds[plane_one]; i++) {
        for (int j = 0; j < bounds[plane_two]; j++) {
            // Iterate over plane
            int offset = i*strides[plane_one] + j*strides[plane_two];
            plaquette_buffer[replica_offset + plane_index_offset + offset + p] += delta;
        }
    }
}

extern "C" __global__ void sum_winding(int* plaquette_buffer, int* sum_buffer,
        int replicas, int t, int x, int y, int z)
{
    // Sum tx plaquettes along y and z axis.
    // Start off similar to global updates.
    // tx : 1  (sum on yz)
    // ty : 1  (sum on xz)
    // tz : 1  (sum on xy)
    // xy : 1  (sum on tz)
    // xz : 1  (sum on ty)
    // yz : 1  (sum on tx)
    int num_planes_per_replica = 6;
    int globalThreadNum = get_thread_number();
    if (globalThreadNum >= replicas * num_planes_per_replica) {
        return;
    }

    int replica_index = globalThreadNum / num_planes_per_replica;
    int p = globalThreadNum % num_planes_per_replica;

    int sum_one_arr[] = {2, 1, 1, 0, 0, 0};
    int sum_two_arr[] = {3, 3, 2, 3, 2, 1};

    int plaquettes_per_txyzslice = 6;
    int plaquettes_per_txyslice = z * plaquettes_per_txyzslice;
    int plaquettes_per_txslice = y * plaquettes_per_txyslice;
    int plaquettes_per_tslice = x * plaquettes_per_txslice;
    int strides[] = {plaquettes_per_tslice, plaquettes_per_txslice, plaquettes_per_txyslice, plaquettes_per_txyzslice};
    int bounds[] = {t,x,y,z};

    // Indexed dim one/two are 2 and 3 if dealing with a tx plane
    int sum_dim_one = sum_one_arr[p];
    int sum_dim_two = sum_two_arr[p];

    int replica_offset = replica_index * (t*x*y*z*6);

    int wp = 0;
    for (int i = 0; i < bounds[sum_dim_one]; i++) {
        for (int j = 0; j < bounds[sum_dim_two]; j++) {
            // Iterate over plane
            int offset = i*strides[sum_dim_one] + j*strides[sum_dim_two];
            wp += plaquette_buffer[replica_offset + offset + p];
        }
    }
    sum_buffer[globalThreadNum] = wp;
}


// Sign convention.
const int sign_convention[4][6] = {
    {1, -1, 0, 1, 0, 0},
    {1, 0, -1, 0, 1, 0},
    {0, 1, -1, 0, 0, 1},
    {0, 0, 0, 1, -1, 1}
};
// For each plane type, get the np value and look up potential.
const int planes_for_cube[4][3] = {
    {0, 1, 3},
    {0, 2, 4},
    {1, 2, 5},
    {3, 4, 5}
};
const int normal_dim_for_cube_planeindex[4][3] = {
    {2, 1, 0}, // txy - tx, ty, xy
    {3, 1, 0}, // txz - tx, tz, xz
    {3, 2, 0}, // tyz - ty, tz, yz
    {3, 2, 1}  // xyz - xy, xz, yz
};

extern "C" __global__ void partial_sum_energies(int* plaquette_buffer, float* sum_buffer,
          float* potential_buffer, int* potential_redirect, int potential_vector_size,
          int replicas, int t, int x, int y, int z)
{
    // Go through each of the 6 plaquette types for each site
    // iterate over z and p, thread for each r, t, x, y
    int num_threads = replicas * t * x * y;

    int globalThreadNum = get_thread_number();
    if (globalThreadNum >= num_threads) {
        return;
    }
    // We work in reverse for this.
    // Interleave the replicas (0, 1, 2, 0, 1, 2, ...)
    int replica_index = globalThreadNum % replicas;
    int in_replica_index = globalThreadNum / replicas;
    int replica_offset = replica_index * (t * x * y * z * 6);

    int potential_index = potential_redirect[replica_index];
    int potential_offset = potential_index * potential_vector_size;

    float pot = 0.0;
    for (int i = 0; i < z * 6; i++) {
        int np = plaquette_buffer[replica_offset + (in_replica_index * z * 6) + i];
        pot += potential_buffer[potential_offset + abs(np)];
    }
    sum_buffer[globalThreadNum] = pot;
}

extern "C" __global__ void calculate_edge_sums(int* plaquette_buffer, int* edge_sums_buffer,
          int replicas, int t, int x, int y, int z)
{
    // First -- calculate the index of the plaquette
    int edges_per_txyzslice = 4;
    int edges_per_txyslice = z * edges_per_txyzslice;
    int edges_per_txslice = y * edges_per_txyslice;
    int edges_per_tslice = x * edges_per_txslice;
    int edges_per_replica = t * edges_per_tslice;

    int globalThreadNum = get_thread_number();
    if (globalThreadNum >= replicas * edges_per_replica) {
        return;
    }

    int replica_index = globalThreadNum / edges_per_replica;
    int within_replica_index = globalThreadNum % edges_per_replica;

    int t_index = within_replica_index / edges_per_tslice;
    int within_t_index = within_replica_index % edges_per_tslice;

    int x_index = within_t_index / edges_per_txslice;
    int within_x_index = within_t_index % edges_per_txslice;

    int y_index = within_x_index / edges_per_txyslice;
    int within_y_index = within_x_index % edges_per_txyslice;

    int z_index = within_y_index / edges_per_txyzslice;
    int edge_type = within_x_index % edges_per_txyzslice;

    int build_plaquettes_per_txyzslice = 6;
    int build_plaquettes_per_txyslice = z * build_plaquettes_per_txyzslice;
    int build_plaquettes_per_txslice = y * build_plaquettes_per_txyslice;
    int build_plaquettes_per_tslice = x * build_plaquettes_per_txslice;
    int build_plaquettes_per_replica = t * build_plaquettes_per_tslice;


    int coords_per_replica = t * x * y * z;
    int replica_offset = replica_index * coords_per_replica;
    int coord_index = globalThreadNum / 4;
    int coords[] = {t_index, x_index, y_index, z_index};
    int bounds[] = {t, x, y, z};
    int deltas[] = {build_plaquettes_per_tslice, build_plaquettes_per_txslice, build_plaquettes_per_txyslice, build_plaquettes_per_txyzslice};

    int edge_sum = get_edge_sum_from_plaquettes(plaquette_buffer, edge_type, coord_index, coords, bounds, deltas);

    edge_sums_buffer[globalThreadNum] = edge_sum;
}

extern "C" __global__ void single_local_update_plaquettes(int* plaquette_buffer,
          float* potential_buffer, int* potential_redirect, int potential_vector_size,
          float* rng_buffer,
          unsigned short cube_type, bool offset,
          int replicas, int t, int x, int y, int z)
{
    int globalThreadNum = get_thread_number();

    if (globalThreadNum >= replicas * t * x * y * z / 2) {
        return;
    }

    // First -- calculate the index of the cube we are focused on updating
    int volumes_per_txyzslice = 1;
    int volumes_per_txyslice = z * volumes_per_txyzslice / 2;
    int volumes_per_txslice = y * volumes_per_txyslice;
    int volumes_per_tslice = x * volumes_per_txslice;
    int volumes_per_replica = t * volumes_per_tslice;

    int replica_index = globalThreadNum / volumes_per_replica;
    int within_replica_index = globalThreadNum % volumes_per_replica;

    int t_index = within_replica_index / volumes_per_tslice;
    int within_t_index = within_replica_index % volumes_per_tslice;

    int x_index = within_t_index / volumes_per_txslice;
    int within_x_index = within_t_index % volumes_per_txslice;

    int y_index = within_x_index / volumes_per_txyslice;
    int within_y_index = within_x_index % volumes_per_txyslice;

    int soft_z_index = within_y_index / volumes_per_txyzslice;
    int int_offset = (int) offset;
    int z_index = 2*soft_z_index + (int_offset + t_index + x_index + y_index)%2;

    int coords_per_z = 1;
    int coords_per_yz = z * coords_per_z;
    int coords_per_xyz = y * coords_per_yz;
    int coords_per_txyz = x * coords_per_xyz;
    int coords_per_replica = t * coords_per_txyz;
    int coord_index = t_index * coords_per_txyz + x_index * coords_per_xyz + y_index * coords_per_yz + z_index;
    int replica_offset = replica_index * t * x * y * z;

    int coords_delta[4] = {coords_per_txyz, coords_per_xyz, coords_per_yz, coords_per_z};
    int coords[4] = {t_index, x_index, y_index, z_index};
    int bounds[4] = {t, x, y, z};

    // This starts as potentials, but changes to weights later
    float boltzman_weights[3] = {0.0,0.0,0.0};

    // Calculate potentials into boltzman_weights
    int potential_index = potential_redirect[replica_index];
    int potential_offset = potential_index * potential_vector_size;
    for (int i = 0; i<3; i++) {
        int plaquette_type = planes_for_cube[cube_type][i];
        int normal_dim = normal_dim_for_cube_planeindex[cube_type][i];
        int coord_up_index = calculate_index_up_difference(bounds[normal_dim], coords[normal_dim], coords_delta[normal_dim]) + coord_index;
        int np = plaquette_buffer[replica_offset + coord_index*6 + plaquette_type];
        int np_up = plaquette_buffer[replica_offset + coord_up_index*6 + plaquette_type];
        for (int delta = -1; delta <= 1; delta++) {
            int new_np = np + delta * sign_convention[cube_type][plaquette_type];
            int new_np_up = np_up - delta * sign_convention[cube_type][plaquette_type];
            boltzman_weights[delta+1] += potential_buffer[potential_offset + abs(new_np)];
            boltzman_weights[delta+1] += potential_buffer[potential_offset + abs(new_np_up)];
        }
    }

    float min_potential = min(min(boltzman_weights[0], boltzman_weights[1]), boltzman_weights[2]);
    float total_weight = 0.0;
    for (int i = 0; i < 3; i++) {
        boltzman_weights[i] = exp(-boltzman_weights[i] + min_potential);
        total_weight += boltzman_weights[i];
    }

    // Now they are boltzman weights

    float rng = rng_buffer[globalThreadNum] * total_weight;
    int j;
    for (j = 0; j < 3; j++) {
        rng -= boltzman_weights[j];
        if (rng <= 0.0) {
            break;
        }
    }

    int delta = j-1;

    for (int i = 0; i<3; i++) {
        int plaquette_type = planes_for_cube[cube_type][i];
        int normal_dim = normal_dim_for_cube_planeindex[cube_type][i];
        int coord_up_index = calculate_index_up_difference(bounds[normal_dim], coords[normal_dim], coords_delta[normal_dim]) + coord_index;
        int np = plaquette_buffer[replica_offset + coord_index*6 + plaquette_type];
        int np_up = plaquette_buffer[replica_offset + coord_up_index*6 + plaquette_type];
        int new_np = np + delta * sign_convention[cube_type][plaquette_type];
        int new_np_up = np_up - delta * sign_convention[cube_type][plaquette_type];

        plaquette_buffer[replica_offset + coord_index*6 + plaquette_type] = new_np;
        plaquette_buffer[replica_offset + coord_up_index*6 + plaquette_type] = new_np_up;
    }
}