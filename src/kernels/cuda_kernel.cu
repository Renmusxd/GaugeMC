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

__device__ int sign(float x)
{
	int t = x<0 ? -1 : 0;
	return x > 0 ? 1 : t;
}

__device__ int get_edge_sum_from_plaquettes(int* plaquette_buffer, unsigned short edge_type,
        int coord_index, int* coords, int* bounds, int* coord_deltas)
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
        int down_index = calculate_index_down_difference(bounds[other_edge], coords[other_edge], 6*coord_deltas[other_edge]) + plaquette_index;

        int down_diff = plaquette_buffer[plaquette_index] - plaquette_buffer[down_index];

        int sign = 1;
        if (edge_type > other_edge) {
            sign = -1;
        }

        sum += sign * down_diff;
    }

    return sum;
}

__device__ int get_coords_from_index(int index, int* bounds, int* output_coords) {
    // TODO fix the bug here: index=216 bounds={3,3,6,6} returns 1 and outputs {0, 0, 0, 0};
    int threads_per_replica = bounds[0] * bounds[1] * bounds[2] * bounds[3];
    int replica_index = index / threads_per_replica;
    int remaining = index % threads_per_replica;

    int threads_per_t = bounds[1] * bounds[2] * bounds[3];
    output_coords[0] = remaining / threads_per_t;
    remaining = remaining % threads_per_t;

    int threads_per_x = bounds[2] * bounds[3];
    output_coords[1] = remaining / threads_per_x;
    remaining = remaining % threads_per_x;

    int threads_per_y = bounds[3];
    output_coords[2] = remaining / threads_per_y;
    output_coords[3] = remaining % threads_per_y;

    return replica_index;
}

const unsigned short edges_a[6] = {0,0,0,1,1,2};
const unsigned short edges_b[6] = {1,2,3,2,3,3};
__device__ void edge_types_for_plaquette_type(unsigned short plaquette_type, unsigned short* edge_a,  unsigned short* edge_b) {
    *edge_a = edges_a[plaquette_type];
    *edge_b = edges_b[plaquette_type];
}

__device__ unsigned short matter_flowing_through_coord(int* plaquette_buffer,
    int replica_and_coord_index,
    int* coords, int* coord_bounds, int* coord_deltas)
{
    unsigned short matter_count = 0;
    for (unsigned short edge_type = 0; edge_type < 4; edge_type++) {
        int up_sum = get_edge_sum_from_plaquettes(plaquette_buffer, edge_type, replica_and_coord_index, coords, coord_bounds, coord_deltas);

        int old_val = coords[edge_type];
        int index_delta = calculate_index_down_difference(coord_bounds[edge_type], coords[edge_type], coord_deltas[edge_type]);
        coords[edge_type] = (coords[edge_type] - 1 + coord_bounds[edge_type]) % coord_bounds[edge_type];
        int down_sum = get_edge_sum_from_plaquettes(plaquette_buffer, edge_type, replica_and_coord_index + index_delta, coords, coord_bounds, coord_deltas);
        coords[edge_type] = old_val;

        matter_count += abs(up_sum) + abs(down_sum);
    }

    return matter_count / 2;
}

/*
Checks the amount of matter flowing through each of the 4 corners of a
plaquette, writes to output number of flows per corner.
*/
__device__ void check_plaquette_matter_corners(int* plaquette_buffer,
    unsigned short plaquette_type,
    int replica_and_coord_index, int* coords, int* coord_bounds, int* coord_deltas,
    unsigned short* output)
{
    unsigned short edge_a = 0;
    unsigned short edge_b = 0;
    edge_types_for_plaquette_type(plaquette_type, &edge_a, &edge_b);

    int edge_a_delta = calculate_index_up_difference(coord_bounds[edge_a], coords[edge_a], coord_deltas[edge_a]);
    int edge_b_delta = calculate_index_up_difference(coord_bounds[edge_b], coords[edge_b], coord_deltas[edge_b]);

    int old_a = coords[edge_a];
    int old_b = coords[edge_b];

    for (unsigned short inc_a = 0; inc_a < 2; inc_a++) {
        for (unsigned short inc_b = 0; inc_b < 2; inc_b++) {
            coords[edge_a] = (old_a + inc_a) % coord_bounds[edge_a];
            coords[edge_b] = (old_b + inc_b) % coord_bounds[edge_b];
            int corner_coord_index = replica_and_coord_index + inc_a*edge_a_delta + inc_b*edge_b_delta;

            output[(inc_a << 1) | inc_b] = matter_flowing_through_coord(plaquette_buffer, corner_coord_index, coords, coord_bounds, coord_deltas);
        }
    }

    coords[edge_a] = old_a;
    coords[edge_b] = old_b;
}

/*
Calculates the violations on each of the 4 edges around a plaquette.
*/
__device__ void get_edge_violations_around_plaquette(int* plaquette_buffer,
    unsigned short plaquette_type,
    int replica_and_coord_index, int* coords, int* coord_bounds, int* coord_deltas,
    int* output)
{
    unsigned short edge_a = 0;
    unsigned short edge_b = 0;
    edge_types_for_plaquette_type(plaquette_type, &edge_a, &edge_b);

    output[0] = get_edge_sum_from_plaquettes(plaquette_buffer, edge_a, replica_and_coord_index, coords, coord_bounds, coord_deltas);
    output[1] = get_edge_sum_from_plaquettes(plaquette_buffer, edge_b, replica_and_coord_index, coords, coord_bounds, coord_deltas);

    int edge_a_delta = calculate_index_up_difference(coord_bounds[edge_a], coords[edge_a], coord_deltas[edge_a]);

    int old_val = coords[edge_a];
    coords[edge_a] = (coords[edge_a] + 1) % coord_bounds[edge_a];
    output[2] = get_edge_sum_from_plaquettes(plaquette_buffer, edge_b, replica_and_coord_index + edge_a_delta, coords, coord_bounds, coord_deltas);
    coords[edge_a] = old_val;

    int edge_b_delta = calculate_index_up_difference(coord_bounds[edge_b], coords[edge_b], coord_deltas[edge_b]);
    old_val = coords[edge_b];
    coords[edge_b] = (coords[edge_b] + 1) % coord_bounds[edge_b];
    output[3] = get_edge_sum_from_plaquettes(plaquette_buffer, edge_a, replica_and_coord_index + edge_b_delta, coords, coord_bounds, coord_deltas);
    coords[edge_b] = old_val;
}

// Calculate min(1.0, exp(-weight)) without overflow.
__device__ float metropolis_prob(float weight) {
    if (weight < 0) {
        return 1.0;
    }
    return exp(-weight);
}

/*
Count the number of flows through a coordinate. 1 in 1 out = 1 flow.
*/
extern "C" __global__ void calculate_matter_flow_through_coords(int* plaquette_buffer, int* matter_flow_buffer,
    int replicas, int t, int x, int y, int z)
{
    int globalThreadNum = get_thread_number();

    if (globalThreadNum >= replicas * t * x * y * z) {
        return;
    }

    int bounds[4] = {t,x,y,z};
    int coords[4] = {0,0,0,0};
    int replica_index = get_coords_from_index(globalThreadNum, bounds, coords);
    int coord_deltas[4] = {x*y*z, y*z, z, 1};
    matter_flow_buffer[globalThreadNum] = matter_flowing_through_coord(plaquette_buffer, globalThreadNum, coords, bounds, coord_deltas);
}

extern "C" __global__ void calculate_matter_corners_for_plaquettes(int* plaquette_buffer, int* matter_corner_buffer,
    unsigned short plaquette_type, int replicas, int t, int x, int y, int z)
{
    int globalThreadNum = get_thread_number();

    if (globalThreadNum >= replicas * t * x * y * z) {
        return;
    }

    int bounds[4] = {t,x,y,z};
    int coords[4] = {0,0,0,0};
    int replica_index = get_coords_from_index(globalThreadNum, bounds, coords);
    int coord_deltas[4] = {x*y*z, y*z, z, 1};

    unsigned short matter[4] = {0,0,0,0};
    check_plaquette_matter_corners(plaquette_buffer, plaquette_type, globalThreadNum, coords, bounds, coord_deltas, matter);

    for (unsigned short i = 0; i < 4; i++) {
        matter_corner_buffer[4*globalThreadNum + i] = matter[i];
    }
}

extern "C" __global__ void wilson_loop_probs(int* plaquette_buffer,
    float* potential_buffer, float* chemical_potential_buffer,
    int* potential_redirect, int potential_vector_size,
    int* aspect_ratios, float* probability_log,
    unsigned short plaquette_type,
    int replicas, int t, int x, int y, int z)
{
    int replica_index = get_thread_number();
    if (replica_index >= replicas) {
        return;
    }

    int aspect_a = aspect_ratios[2*replica_index];
    int aspect_b = aspect_ratios[2*replica_index + 1];

    unsigned short edge_a = 0;
    unsigned short edge_b = 0;
    edge_types_for_plaquette_type(plaquette_type, &edge_a, &edge_b);

    const int coords_per_replica = t*x*y*z;
    int coords_delta[4] = {x*y*z, y*z, z, 1};
    int bounds[4] = {t, x, y, z};

    const int replica_offset = coords_per_replica * replica_index;

    int potential_index = potential_redirect[replica_index];
    int potential_offset = potential_index * potential_vector_size;
    // float base_potential = potential_buffer[potential_offset + abs(np)] - np * chemical_potential_buffer[potential_index]

    // First consider +/- edges directions
    int edges[2] = {edge_a, edge_b};
    int aspects[2] = {aspect_a, aspect_b};
    float inc_costs[2] = {0.0, 0.0};
    float dec_costs[2] = {0.0, 0.0};
    for (int edge = 0; edge < 2; edge++) {
        // We are pushing/pulling in the edges[edge] directions.
        // We iterate along the edges[1-edge] direction.
        // We will not consider wrapping, since we are working with a square with a corner at (0,0) growing in the positive
        // directions. So our coord deltas do not need fancy stuff.
        int para_aspect = aspects[edge];
        int para_delta = coords_delta[edges[edge]];
        int ortho_aspect = aspects[1-edge];
        int ortho_delta = coords_delta[edges[1-edge]];

        for (int ortho_index = 0; ortho_index < ortho_aspect; ortho_index++) {
            // Calculate cost of extending loop in the para direction.
            int inc_index = (para_aspect + 1) * para_delta + ortho_index * ortho_delta + plaquette_type;
            int np = plaquette_buffer[inc_index];
            inc_costs[edge] += potential_buffer[potential_offset + abs(np+1)] - potential_buffer[potential_offset + abs(np)];

            // Calculate cost of shrinking loop downwards
            int dec_index = para_aspect * para_delta + ortho_index * ortho_delta + plaquette_type;
            np = plaquette_buffer[dec_index];
            dec_costs[edge] += potential_buffer[potential_offset + abs(np-1)] - potential_buffer[potential_offset + abs(np)];
        }
        inc_costs[edge] -= ortho_aspect * chemical_potential_buffer[potential_index];
        dec_costs[edge] += ortho_aspect * chemical_potential_buffer[potential_index];
    }
    // Now calculate the inc corner for when both edges are pushed at once.
    int inc_index = (aspects[0] + 1) * coords_delta[edges[0]] + (aspects[1] + 1) * coords_delta[edges[1]] + plaquette_type;
    int np = plaquette_buffer[inc_index];
    float corner_inc_cost = potential_buffer[potential_offset + abs(np+1)] - potential_buffer[potential_offset + abs(np)] - chemical_potential_buffer[potential_index];

    // And calculate the dec corner which was double counted if each edge is pulled at once.
    int dec_index = aspects[0] * coords_delta[edges[0]] + aspects[1] * coords_delta[edges[1]] + plaquette_type;
    np = plaquette_buffer[dec_index];
    float corner_dec_cost = potential_buffer[potential_offset + abs(np-1)] - potential_buffer[potential_offset + abs(np)] + chemical_potential_buffer[potential_index];

    // Now write the probabilities of various moves into the output buffer

    // Decrease edge_a
    probability_log[0] += metropolis_prob(dec_costs[0]);
    // Decrease edge_b
    probability_log[1] += metropolis_prob(dec_costs[1]);
    // Decrease edge_a and edge_b (taking into account the double counted corner).
    probability_log[2] += metropolis_prob(dec_costs[0] + dec_costs[1] - corner_dec_cost);
    // Increase edge_a
    probability_log[3] += metropolis_prob(inc_costs[0]);
    // Increase edge_b
    probability_log[4] += metropolis_prob(inc_costs[1]);
    // Increase edge_a and edge_b (taking into account the uncounted corner).
    probability_log[5] += metropolis_prob(inc_costs[0] + inc_costs[1] + corner_inc_cost);
}

extern "C" __global__ void update_matter_loops(int* plaquette_buffer,
    float* potential_buffer, float* chemical_potential_buffer, int* potential_redirect, int potential_vector_size,
    float* rng_buffer, unsigned short plaquette_type_and_offset, // unsigned short plaquette_type, int offset,
    int replicas, int t, int x, int y, int z)
{
    const int globalThreadNum = get_thread_number();

    // Number of plaquettes to update is 1/4 of sites.
    //  XO     each of 4
    //  OO     in tx plane.
    // We will return early if not in correct place.
    if (globalThreadNum >= replicas * t * x * y * z / 4) {
        return;
    }

    unsigned short plaquette_type = (plaquette_type_and_offset >> 2) & 0b111;
    unsigned short offset = plaquette_type_and_offset & 0b11;

    // Plaquette types:
    unsigned short edge_a = 0;
    unsigned short edge_b = 0;
    edge_types_for_plaquette_type(plaquette_type, &edge_a, &edge_b);

    // First -- calculate the index of the cube we are focused on updating
    int divs[4] = {t,x,y,z};
    divs[edge_a] /= 2;
    divs[edge_b] /= 2;

    int coords[4] = {0, 0, 0, 0};
    int replica_index = get_coords_from_index(globalThreadNum, divs, coords);

    // Adjust offsets
    coords[edge_a] = 2*coords[edge_a] + (offset & 1);
    coords[edge_b] = 2*coords[edge_b] + ((offset>>1)&1);

    const int coords_per_replica = t*x*y*z;
    int coords_delta[4] = {x*y*z, y*z, z, 1};
    int bounds[4] = {t, x, y, z};

    const int coord_and_replica_index = replica_index*coords_per_replica + coords_delta[0]*coords[0] + coords_delta[1]*coords[1] + coords_delta[2]*coords[2] + coords_delta[3]*coords[3];

    int plaquette_violations[4] = {0,0,0,0};
    get_edge_violations_around_plaquette(plaquette_buffer, plaquette_type, coord_and_replica_index, coords, bounds, coords_delta, plaquette_violations);

    bool any_above_one = false;
    unsigned short count_of_ones = 0;
    for (int i = 0; i<4; i++) {
        any_above_one |= abs(plaquette_violations[i]) > 1;
        // Since all violations are enforced to be 0 or 1, this counts 1s.
        count_of_ones += abs(plaquette_violations[i]);
    }
    if (any_above_one || count_of_ones==0 || count_of_ones==4) {
        return;
    }

    int violation_location = 0;
    for (; violation_location < 4; violation_location++) {
        if (plaquette_violations[violation_location] != 0) {
            break;
        }
    }

    // Special check of count_of_ones==2
    if (count_of_ones == 2) {
        // If the 1s are on opposite sides, return early to not close loops.
        const unsigned short spot_to_check[4] = {3, 2, 1, 0};
        unsigned short potential_violation_location = spot_to_check[violation_location];
        if (abs(plaquette_violations[violation_location]) == abs(plaquette_violations[potential_violation_location])) {
            return;
        }
    }


    int violation_at_loc = plaquette_violations[violation_location];

    const int inc_incurrs[4] = {1, -1, 1, -1};
    // violation_at_loc + plaquette_delta * inc_incurrs = 0
    int plaquette_delta = - sign(violation_at_loc) / sign(inc_incurrs[violation_location]);

    int np = plaquette_buffer[6*coord_and_replica_index + plaquette_type];

    int potential_index = potential_redirect[replica_index];
    int potential_offset = potential_index * potential_vector_size;

    float base_potential = potential_buffer[potential_offset + abs(np)]; // - np * chemical_potential_buffer[potential_index]
    float new_potential = potential_buffer[potential_offset + abs(np+plaquette_delta)];  // - (np+plaquette_delta) * chemical_potential_buffer[potential_index]
    float potential_diff = new_potential - base_potential - plaquette_delta * chemical_potential_buffer[potential_index];
    float min_pot = min(0.0, potential_diff);

    float stay_weight = exp( - (0.0 - min_pot) );
    float inc_weight = exp( - (potential_diff - min_pot) );

    float rng = rng_buffer[globalThreadNum] * (stay_weight + inc_weight);

    if (rng > stay_weight) {
        // Now check if we made any crossing matter lines.
        plaquette_buffer[6*coord_and_replica_index + plaquette_type] += plaquette_delta;

        int np = plaquette_buffer[6*coord_and_replica_index + plaquette_type];

        // Now check if we made any crossing matter lines.
        unsigned short matter_flows[4] = {0,0,0,0};
        check_plaquette_matter_corners(plaquette_buffer, plaquette_type, coord_and_replica_index, coords, bounds, coords_delta, matter_flows);

        bool any_cross_flows = false;
        for (int i = 0; i < 4; i++) {
            any_cross_flows |= (matter_flows[i] > 1);
        }
        // If we cross flows, undo change.
        if (any_cross_flows) {
            plaquette_buffer[6*coord_and_replica_index + plaquette_type] -= plaquette_delta;
        }
    }
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
    int plane_area_per_type[] = {t*x, t*y, t*z, x*y, x*z, y*z};
    boltzman_weights[0] += chemical_potential_buffer[potential_index] * plane_area_per_type[p];
    boltzman_weights[2] -= chemical_potential_buffer[potential_index] * plane_area_per_type[p];

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
          float* potential_buffer, float* chemical_potential_buffer, int* potential_redirect, int potential_vector_size,
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
        pot -= np * chemical_potential_buffer[potential_index];
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

    int coords_per_z = 1;
    int coords_per_y = z * coords_per_z;
    int coords_per_x = y * coords_per_y;
    int coords_per_t = x * coords_per_x;
    int coords_per_replica = t * coords_per_t;
    int replica_offset = replica_index * coords_per_replica;

    int coord_index = globalThreadNum / 4;
    int coords[4] = {t_index, x_index, y_index, z_index};
    int bounds[4] = {t, x, y, z};
    int coord_deltas[4] = {coords_per_t, coords_per_x, coords_per_y, coords_per_z};

    int edge_sum = get_edge_sum_from_plaquettes(plaquette_buffer, edge_type, coord_index, coords, bounds, coord_deltas);

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
    int replica_offset = replica_index * coords_per_replica * 6;

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
            // We dont need chemical potential since we always add as many even as odd increments.
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