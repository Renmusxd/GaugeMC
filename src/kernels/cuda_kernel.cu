// int __device__ calculate_cube_index(int t, int x, int y, int z,
//     int replica_index, int t_index, int x_index, int y_index, int z_index,
//     unsigned short volume_type)
// {
//         int build_volumes_per_txyzslice = 4;
//         int build_volumes_per_txyslice = z * build_volumes_per_txyzslice;
//         int build_volumes_per_txslice = y * build_volumes_per_txyslice;
//         int build_volumes_per_tslice = x * build_volumes_per_txslice;
//         int build_volumes_per_replica = t * build_volumes_per_tslice;
//
//         int cube_index = replica_index * build_volumes_per_replica;
//         cube_index += t_index * build_volumes_per_tslice;
//         cube_index += x_index * build_volumes_per_txslice;
//         cube_index += y_index * build_volumes_per_txyslice;
//         cube_index += z_index * build_volumes_per_txyzslice;
//         cube_index += volume_type;
//
//         return cube_index;
// }


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

int __device__ cube_mask_to_cube_type(int mask) {
    switch(mask) {
        case 0b1110:
            return 0;
        case 0b1101:
            return 1;
        case 0b1011:
            return 2;
        case 0b0111:
            return 3;
    }
    return -1;
}

int __device__ edge_mask_to_edge_type(int mask) {
    switch(mask) {
        case 0b1000:
            return 0;
        case 0b0100:
            return 1;
        case 0b0010:
            return 2;
        case 0b0001:
            return 3;
    }
    return -1;
}

__device__ void print_cube_info(int t, int x, int y, int z, int index) {
    int build_volumes_per_txyzslice = 4;
    int build_volumes_per_txyslice = z * build_volumes_per_txyzslice;
    int build_volumes_per_txslice = y * build_volumes_per_txyslice;
    int build_volumes_per_tslice = x * build_volumes_per_txslice;
    int build_volumes_per_replica = t * build_volumes_per_tslice;

    int r_index = index / build_volumes_per_replica;
    index = index % build_volumes_per_replica;

    int t_index = index / build_volumes_per_tslice;
    index = index % build_volumes_per_tslice;

    int x_index = index / build_volumes_per_txslice;
    index = index % build_volumes_per_txslice;

    int y_index = index / build_volumes_per_txyslice;
    index = index % build_volumes_per_txyslice;

    int z_index = index / build_volumes_per_txyzslice;
    int cube_type = index % build_volumes_per_txyzslice;

    printf("(%d, %d, %d, %d, %d, %d)", r_index, t_index, x_index, y_index, z_index, cube_type);
}

__device__ void memcpy(int* src, int* dest, int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = src[i];
    }
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

        sum += sign * down_diff; // TODO fix sign
    }

    return sum;
}


// Sign convention.
const int sign_convention[4][6] = {
    {1, -1, 0, 1, 0, 0},
    {1, 0, -1, 0, 1, 0},
    {0, 1, -1, 0, 0, 1},
    {0, 0, 0, 1, -1, 1}
};

// Let mu=plane_axis_one and nu=plane_axis_two
// global_index is the index for cube 0 at coords.
__device__ int get_np_from_state(int* state_buffer,
        int plane_axis_one, int plane_axis_two,
        int coord_index, int* coords, int* bounds, int* cube_deltas)
{
    if (plane_axis_one == plane_axis_two) {
        return ~0;
    }
    int mask = ((1 << (3 - plane_axis_one)) | (1 << (3 - plane_axis_two)));
    int free_edge_one = 0;
    int free_edge_two = 0;
    int cube_type = 0;
    int other_cube_type = 0;
    int plaquette_type = 0;

    // Cube types
    // 0: txy -- 1110 (+1)
    // 1: txz -- 1101 (-1)
    // 2: tyz -- 1011 (-1)
    // 3: xyz -- 0111 (+1)
    switch(mask) {
        // yz
        case 0b0011:
            free_edge_one = 0;
            free_edge_two = 1;
            // Cube types are tyz and xyz
            cube_type = 2;
            other_cube_type = 3;
            plaquette_type = 5;
            break;
        // xz
        case 0b0101:
            free_edge_one = 0;
            free_edge_two = 2;
            // Cube types are txz and xyz
            cube_type = 1;
            other_cube_type = 3;
            plaquette_type = 4;
            break;
        // xy
        case 0b0110:
            free_edge_one = 0;
            free_edge_two = 3;
            // Cube types are txy and xyz
            cube_type = 0;
            other_cube_type = 3;
            plaquette_type = 3;
            break;
        // tz
        case 0b1001:
            free_edge_one = 1;
            free_edge_two = 2;
            // Cube types are txz and tyz
            cube_type = 1;
            other_cube_type = 2;
            plaquette_type = 2;
            break;
        // ty
        case 0b1010:
            free_edge_one = 1;
            free_edge_two = 3;
            // Cube types are txy and tyz
            cube_type = 0;
            other_cube_type = 2;
            plaquette_type = 1;
            break;
        // tx
        case 0b1100:
            free_edge_one = 2;
            free_edge_two = 3;
            // Cube types are txy and txz
            cube_type = 0;
            other_cube_type = 1;
            plaquette_type = 0;
            break;
        default:
            return ~0;
    }

    int global_index = coord_index * 4;
    int cube_index = global_index + cube_type;
    int down_index = calculate_index_down_difference(bounds[free_edge_one], coords[free_edge_one], cube_deltas[free_edge_one]) + cube_index;
    int down_diff = state_buffer[cube_index] - state_buffer[down_index];

    int other_cube_index = global_index + other_cube_type;
    int other_down_index = calculate_index_down_difference(bounds[free_edge_two], coords[free_edge_two], cube_deltas[free_edge_two]) + other_cube_index;
    int other_down_diff = state_buffer[other_cube_index] - state_buffer[other_down_index];

    return sign_convention[cube_type][plaquette_type] * down_diff + sign_convention[other_cube_type][plaquette_type] * other_down_diff;

    // Signs using e_{abcd}
    // n_{ab} = e_{abcd} d_c m_d = e_{abcd} d_c m_{abc}
    // e_{txyz} = 1
    // e_{txzy} = -1
    // e_{tyzx} = -1
    // e_{xyzt} = 1
    // int signs[] = {1, -1, -1, 1}; // TODO check consistency?
    // return global_sign * (signs[cube_type] * down_diff + signs[other_cube_type] * other_down_diff);
}


// For each plane type, get the np value and look up potential.
const int planes_for_cube[4][3] = {
    {0, 1, 3},
    {0, 2, 4},
    {1, 2, 5},
    {3, 4, 5}
};
const int dims_for_plane_type[6][2] = {
    {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
};
const int normal_dim_for_cube_planeindex[4][3] = {
    {2, 1, 0}, // txy - tx, ty, xy
    {3, 1, 0}, // txz - tx, tz, xz
    {3, 2, 0}, // tyz - ty, tz, yz
    {3, 2, 1}  // xyz - xy, xz, yz
};
__device__ void get_dec_stay_inc_potentials(int* state_buffer,
          float* potential_buffer, int potential_vector_size,
          int cube_type, int coord_index, int replica_index,
          int* coords, int* bounds, int* cube_deltas, float* potential_changes)
{
    int yz_delta = coords[3];
    int xyz_delta = coords[2] * yz_delta;
    int txyz_delta = coords[1] * xyz_delta;
    int coords_delta[4] = {txyz_delta, xyz_delta, yz_delta, 1};

    int cube_index = coord_index*4 + cube_type;
    int potential_offset = potential_vector_size * replica_index;

    for (int i = 0; i < 3; i++) {
        int plane_type = planes_for_cube[cube_type][i];
        int dim_a = dims_for_plane_type[plane_type][0];
        int dim_b = dims_for_plane_type[plane_type][1];
        int remaining_dim = normal_dim_for_cube_planeindex[cube_type][i];

        int starting_value = state_buffer[cube_index];
        for (int delta = -1; delta < 2; delta++) {
            state_buffer[cube_index] = starting_value + delta;
            int plane_np = get_np_from_state(state_buffer, dim_a, dim_b, coord_index, coords, bounds, cube_deltas);
            potential_changes[delta+1] += potential_buffer[potential_offset + abs(plane_np)];
        }

        int coord_up_index = calculate_index_up_difference(bounds[remaining_dim], coords[remaining_dim], coords_delta[remaining_dim]) + coord_index;
        for (int delta = -1; delta < 2; delta++) {
            state_buffer[cube_index] = starting_value + delta;
            int plane_np = get_np_from_state(state_buffer, dim_a, dim_b, coord_up_index, coords, bounds, cube_deltas);
            potential_changes[delta+1] += potential_buffer[potential_offset + abs(plane_np)];
        }
        state_buffer[cube_index] = starting_value;
    }
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

    int coord_index = (globalThreadNum / 4);
    int coords[] = {t_index, x_index, y_index, z_index};
    int bounds[] = {t, x, y, z};
    int deltas[] = {build_plaquettes_per_tslice, build_plaquettes_per_txslice, build_plaquettes_per_txyslice, build_plaquettes_per_txyzslice};

    int edge_sum = get_edge_sum_from_plaquettes(plaquette_buffer, edge_type, coord_index, coords, bounds, deltas);
    edge_sums_buffer[globalThreadNum] = edge_sum;
}

extern "C" __global__ void calculate_plaquettes(int* state_buffer, int* plaquette_buffer,
          int replicas, int t, int x, int y, int z)
{
    int globalThreadNum = get_thread_number();
    if (globalThreadNum >= replicas * t * x * y * z * 6) {
        return;
    }

    // First -- calculate the index of the plaquette
    int plaquettes_per_txyzslice = 6;
    int plaquettes_per_txyslice = z * plaquettes_per_txyzslice;
    int plaquettes_per_txslice = y * plaquettes_per_txyslice;
    int plaquettes_per_tslice = x * plaquettes_per_txslice;
    int plaquettes_per_replica = t * plaquettes_per_tslice;

    int replica_index = globalThreadNum / plaquettes_per_replica;
    int within_replica_index = globalThreadNum % plaquettes_per_replica;

    int t_index = within_replica_index / plaquettes_per_tslice;
    int within_t_index = within_replica_index % plaquettes_per_tslice;

    int x_index = within_t_index / plaquettes_per_txslice;
    int within_x_index = within_t_index % plaquettes_per_txslice;

    int y_index = within_x_index / plaquettes_per_txyslice;
    int within_y_index = within_x_index % plaquettes_per_txyslice;

    int z_index = within_y_index / plaquettes_per_txyzslice;
    int plaquette_type = within_x_index % plaquettes_per_txyzslice;

    int plane_axis_one = 0;
    int plane_axis_two = 0;
    switch(plaquette_type) {
        case 0:
            plane_axis_one = 0;
            plane_axis_two = 1;
            break;
        case 1:
            plane_axis_one = 0;
            plane_axis_two = 2;
            break;
        case 2:
            plane_axis_one = 0;
            plane_axis_two = 3;
            break;
        case 3:
            plane_axis_one = 1;
            plane_axis_two = 2;
            break;
        case 4:
            plane_axis_one = 1;
            plane_axis_two = 3;
            break;
        case 5:
            plane_axis_one = 2;
            plane_axis_two = 3;
            break;
    }


    int build_volumes_per_txyzslice = 4;
    int build_volumes_per_txyslice = z * build_volumes_per_txyzslice;
    int build_volumes_per_txslice = y * build_volumes_per_txyslice;
    int build_volumes_per_tslice = x * build_volumes_per_txslice;
    int build_volumes_per_replica = t * build_volumes_per_tslice;

    int coord_index = (globalThreadNum / 6);
    int coords[] = {t_index, x_index, y_index, z_index};
    int bounds[] = {t, x, y, z};
    int deltas[] = {build_volumes_per_tslice, build_volumes_per_txslice, build_volumes_per_txyslice, build_volumes_per_txyzslice};

    int np = get_np_from_state(state_buffer, plane_axis_one, plane_axis_two, coord_index, coords, bounds, deltas);
    plaquette_buffer[globalThreadNum] = np;
}

extern "C" __global__ void single_local_update_cubes(int* state_buffer,
          float* potential_buffer, int potential_vector_size,
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
    int z_index = 2*soft_z_index + (int_offset + (t_index + x_index + y_index)%2);

    int coords_per_z = 1;
    int coords_per_yz = z * coords_per_z;
    int coords_per_xyz = y * coords_per_yz;
    int coords_per_txyz = x * coords_per_xyz;
    int coord_index = t_index * coords_per_txyz + x_index * coords_per_xyz + y_index * coords_per_yz + z_index;

    int cubes_per_z = 4;
    int cubes_per_yz = z * cubes_per_z;
    int cubes_per_xyz = y * cubes_per_yz;
    int cubes_per_txyz = x * cubes_per_xyz;
    int cube_deltas[4] = {cubes_per_txyz, cubes_per_xyz, cubes_per_yz, cubes_per_z};
    int coords[4] = {t_index, x_index, y_index, z_index};
    int bounds[4] = {t, x, y, z};

    // This starts as potentials, but changes to weights later
    float boltzman_weights[3] = {0.0,0.0,0.0};
    get_dec_stay_inc_potentials(state_buffer, potential_buffer, potential_vector_size,
            cube_type, coord_index, replica_index,
            coords, bounds, cube_deltas,
            boltzman_weights);
    float min_potential = min(min(boltzman_weights[0], boltzman_weights[1]), boltzman_weights[2]);
    float total_weight = 0.0;
    for (int i = 0; i < 3; i++) {
        boltzman_weights[i] -= exp(-boltzman_weights[i] + min_potential);
        total_weight += boltzman_weights[i];
    }
    // Now they are boltzman weights

    float rng = rng_buffer[globalThreadNum] * total_weight;
    int i = 0;
    for (; i < 3; i++) {
        rng -= boltzman_weights[i];
        if (rng <= 0.0) {
            break;
        }
    }

    int delta = i-1;
    state_buffer[coord_index*4 + cube_type] += delta;
}


extern "C" __global__ void single_local_update_plaquettes(int* plaquette_buffer,
          float* potential_buffer, int potential_vector_size,
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
    int z_index = 2*soft_z_index + (int_offset + (t_index + x_index + y_index)%2);

    int coords_per_z = 1;
    int coords_per_yz = z * coords_per_z;
    int coords_per_xyz = y * coords_per_yz;
    int coords_per_txyz = x * coords_per_xyz;
    int coord_index = t_index * coords_per_txyz + x_index * coords_per_xyz + y_index * coords_per_yz + z_index;

    int coords_delta[4] = {coords_per_txyz, coords_per_xyz, coords_per_yz, coords_per_z};
    int coords[4] = {t_index, x_index, y_index, z_index};
    int bounds[4] = {t, x, y, z};

    // This starts as potentials, but changes to weights later
    float boltzman_weights[3] = {0.0,0.0,0.0};

    // Calculate potentials into boltzman_weights
    int potential_offset = replica_index * potential_vector_size;
    for (int i = 0; i<3; i++) {
        int plaquette_type = planes_for_cube[cube_type][i];
        int normal_dim = normal_dim_for_cube_planeindex[cube_type][i];
        int coord_up_index = calculate_index_up_difference(bounds[normal_dim], coords[normal_dim], coords_delta[normal_dim]) + coord_index;
        int np = plaquette_buffer[coord_index*6 + plaquette_type];
        int np_up = plaquette_buffer[coord_up_index*6 + plaquette_type];
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
        int np = plaquette_buffer[coord_index*6 + plaquette_type];
        int np_up = plaquette_buffer[coord_up_index*6 + plaquette_type];
        int new_np = np + delta * sign_convention[cube_type][plaquette_type];
        int new_np_up = np_up - delta * sign_convention[cube_type][plaquette_type];
        plaquette_buffer[coord_index*6 + plaquette_type] = new_np;
        plaquette_buffer[coord_up_index*6 + plaquette_type] = new_np_up;
    }
}