
struct State {
  state : array<i32>;
};

struct Vn {
    vn : array<f32>;
};

struct DimInformation {
    // First the dimensions of the graph
    // t, x, y, z, num_replicas,
    // Then the cube selections (first 3 are 3D cube, last is remaining)
    // mu, nu, sigma, rho, offset
    data : array<u32>;
};

struct PCGState {
    state : array<u32>;
};

struct SumBuffer {
    buff : array<f32>;
};

[[group(0), binding(0)]]
var<storage, read_write> state : State;
[[group(0), binding(1)]]
var<storage, read> vn : Vn;
[[group(0), binding(2)]]
var<storage, read> dim_indices : DimInformation;
[[group(0), binding(3)]]
var<storage, read_write> pcgstate : PCGState;
[[group(0), binding(4)]]
var<storage, read_write> sumbuffer : SumBuffer;

fn p_from_dims(first: u32, second: u32) -> u32 {
    // first and second are between 0x00 and 0x11
    // outputs from 0x000 to 0x101
    // This logic made by Karnaugh map with invalid entries replaced by useful values.
    let a = ((first == 1u) | (first == 2u)) & (second == 3u);
    let b = ((first == 0u) & (second == 3u)) | ((first == 1u) & (second == 2u));
    let c = (first == 2u) | (second == 2u);
    return (u32(a) << 2u) | (u32(b) << 1u) | u32(c);
}

fn dims_from_p(p: u32) -> vec2<u32> {
    // t t t x x y
    // x y z y z z
    var v = vec2<u32>(0u, 1u);
    // v[0] = select(0,0,p<3);
    v[0] = select(0u,1u,p<5u);
    v[0] = select(v[0],2u,p==5u);

    // v[1] = select(v[1],1,p==0);
    v[1] = select(v[1],2u,p==1u);
    v[1] = select(v[1],3u,p==2u);
    v[1] = select(v[1],2u,p==3u);
    v[1] = select(v[1],3u,p==4u);
    v[1] = select(v[1],3u,p==5u);

    return v;
}

// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
fn pcg_hash(inp: u32) -> u32 {
    let state : u32 = inp * 747796405u + 2891336453u;
    let word : u32 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn prng(index: u32) -> f32 {
    pcgstate.state[index] = pcg_hash(pcgstate.state[index]);
    // u32 to i32, cast to f32 and divide by 2^31, then shift to 0-1
    //let random_float: f32 = f32(i32(pcgstate.state[global_id.x])) * 2.32830643653869628906e-010 + 0.5;
    // Or convert to f32 and divide by total range.
    return f32(pcgstate.state[index]) / f32(0xffffffffu);
}

[[stage(compute), workgroup_size(256,1,1)]]
fn rotate_pcg([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    var index = global_id.x;

    let num_pcgs = dim_indices.data[0];
    let use_offset = dim_indices.data[1];

    let pcg_index = 2u*index + use_offset;
    if (pcg_index >= num_pcgs) {
        return;
    }
    pcgstate.state[pcg_index] = pcgstate.state[pcg_index] ^ pcgstate.state[(pcg_index + 1u) % num_pcgs];
}

[[stage(compute), workgroup_size(256,1,1)]]
fn initial_sum_energy([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    var index = global_id.x;

    let t = dim_indices.data[0];
    let x = dim_indices.data[1];
    let y = dim_indices.data[2];
    let z = dim_indices.data[3];
    let p = 6u;
    let num_replicas = dim_indices.data[4];

    // Since each dimension starts as an even number, and there are 6 faces, we knows at least
    // 2^4 * 6 entries in each replica.
    // We will just do 16 of those since parallelization is good too.
    // We wont, however, do the reduction in any particular order.
    let initial_reduction = 16u;

    // We group 4 at a time to start.
    let entries_per_replica = (t*x*y*z*p)/initial_reduction;
    let num_threads = entries_per_replica*num_replicas;
    if (index >= num_threads) {
        return;
    }

    let replica_index = index % num_replicas;
    index = index / num_replicas;

    let vn_offset = i32(dim_indices.data[5u + replica_index]);

    let replica_offset = replica_index * (t*x*y*z*p);

    var energy = 0.0;
    for (var i = 0u; i < initial_reduction; i = i+1u) {
        let n = state.state[replica_offset + initial_reduction*index + i];
        energy = energy + vn.vn[vn_offset + abs(n)];
    }

    sumbuffer.buff[global_id.x] = energy;
}


[[stage(compute), workgroup_size(256,1,1)]]
fn incremental_sum_energy([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    var index = global_id.x;

    let num_replicas = dim_indices.data[4];
    // data[5] is for replica 0, ...
    let size_in_replica = dim_indices.data[5u + num_replicas];

    let threads_per_replica = size_in_replica / 2u + size_in_replica % 2u;
    let num_threads = threads_per_replica*num_replicas;
    if (index >= num_threads) {
        return;
    }

    let replica_index = index % num_replicas;
    index = index / num_replicas;

    // Want to fold ith with (N-i)th - reduces by 2 each time and keeps arrangement of replicas.
    let fold_index = (size_in_replica - 1u - index)*num_replicas + replica_index;
    let index = (index * num_replicas) + replica_index;

    if (fold_index == index) {
        return;
    }

    let energy = sumbuffer.buff[index] + sumbuffer.buff[fold_index];
    sumbuffer.buff[index] = energy;
    sumbuffer.buff[fold_index] = 0.0;
}

[[stage(compute), workgroup_size(256,1,1)]]
fn main_global([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    var index = global_id.x;

    let t = dim_indices.data[0];
    let x = dim_indices.data[1];
    let y = dim_indices.data[2];
    let z = dim_indices.data[3];
    let p = 6u;
    let num_replicas = dim_indices.data[4];

    let planes_per_replica = t*(x+y+z) + x*(y+z) + y*z;
    let num_planes = num_replicas*planes_per_replica;

    if (index >= num_planes) {
        return;
    }

    let replica_index = index / planes_per_replica;
    let replica_offset = replica_index * (t*x*y*z*p);
    index = index % planes_per_replica;

    let vn_offset = i32(dim_indices.data[5u + replica_index]);

    // We will be looking at all entries of the form:
    // index = tXYZP + xYZP + yZP + zP + p
    // where two of [t,x,y,z] are variable.
    // index = v1*k1 + v2*k2 + offset
    var k1 = 0u;
    var v1_max = 0u;
    var k2 = 0u;
    var v2_max = 0u;
    var offset = 0u;
    if (index < t*x) {
        let plane = index;
        // Y/Z plane, p=5
        let t_index = plane / x;
        let x_index = plane % x;
        offset = t_index * (x*y*z*p) + x_index * (y*z*p) + 5u;
        v1_max = y;
        v2_max = z;
        k1 = z*p;
        k2 = p;
    } else if (index < t*(x+y)) {
        let plane = index - t*x;
        // X/Z plane, p=4
        let t_index = plane / y;
        let y_index = plane % y;
        offset = t_index * (x*y*z*p) + y_index * (z*p) + 4u;
        v1_max = x;
        v2_max = z;
        k1 = y*z*p;
        k2 = p;
    } else if (index < t*(x+y+z)) {
        let plane = index - t*(x+y);
        // X/Y plane, p=3
        let t_index = plane / z;
        let z_index = plane % z;
        offset = t_index * (x*y*z*p) + z_index * p + 3u;
        v1_max = x;
        v2_max = y;
        k1 = y*z*p;
        k2 = z*p;
    } else if (index < t*(x+y+z) + x*y) {
        let plane = index - t*(x+y+z);
        // T/Z plane p=2
        let x_index = plane / y;
        let y_index = plane % y;
        offset = x_index * (y*z*p) + y_index * (z*p) + 2u;
        v1_max = t;
        v2_max = z;
        k1 = x*y*z*p;
        k2 = p;
    } else if (index < t*(x+y+z) + x*(y+z)) {
        let plane = index - t*(x+y+z) - x*y;
        // T/Y plane p=1
        let x_index = plane / z;
        let z_index = plane % z;
        offset = x_index * (y*z*p) + z_index * p + 1u;
        v1_max = t;
        v2_max = y;
        k1 = x*y*z*p;
        k2 = z*p;
    } else {
        let plane = index - t*(x+y+z) - x*(y+z);
        // T/X plane p=0
        let y_index = plane / z;
        let z_index = plane % z;
        offset = y_index * (z*p) + z_index * p + 0u;
        v1_max = t;
        v2_max = x;
        k1 = x*y*z*p;
        k2 = y*z*p;
    }

    var inc_energy_increase = 0.0;
    var dec_energy_increase = 0.0;
    for (var v1 = 0u; v1 < v1_max; v1 = v1 + 1u) {
        for (var v2 = 0u; v2 < v2_max; v2 = v2 + 1u) {
            let plaquette_index = v1*k1 + v2*k2 + offset;
            let v_at_plaq = state.state[replica_offset + plaquette_index];
            inc_energy_increase = inc_energy_increase + vn.vn[vn_offset + abs(v_at_plaq + 1)] - vn.vn[vn_offset + abs(v_at_plaq)];
            dec_energy_increase = dec_energy_increase + vn.vn[vn_offset + abs(v_at_plaq - 1)] - vn.vn[vn_offset + abs(v_at_plaq)];
        }
    }

    // Shift all choices to the most preferable thing is "0" energy.
    let lowest_e = min(inc_energy_increase, min(dec_energy_increase, 0.0));
    let zero_energy = 0.0 - lowest_e;
    inc_energy_increase = inc_energy_increase - lowest_e;
    dec_energy_increase = dec_energy_increase - lowest_e;

    let add_zer_p = exp(-zero_energy);
    let add_one_p = exp(-inc_energy_increase);
    let sub_one_p = exp(-dec_energy_increase);

    let random_float = prng(global_id.x);
    let random_float = random_float * (add_zer_p + add_one_p + sub_one_p) - add_zer_p;

    if (random_float < 0.0) {
        return;
    }
    let choice = select(-1, 1, random_float < add_one_p);
    for (var v1 = 0u; v1 < v1_max; v1 = v1 + 1u) {
        for (var v2 = 0u; v2 < v2_max; v2 = v2 + 1u) {
            let plaquette_index = v1*k1 + v2*k2 + offset;
            state.state[replica_offset + plaquette_index] = state.state[replica_offset + plaquette_index] + choice;
        }
    }
}

[[stage(compute), workgroup_size(256,1,1)]]
fn main_local([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    var index = global_id.x;

    // Get the bounds.
    let t = dim_indices.data[0];
    let x = dim_indices.data[1];
    let y = dim_indices.data[2];
    let z = dim_indices.data[3];
    let p = 6u;
    let num_replicas = dim_indices.data[4];

    if (global_id.x >= (num_replicas*t*x*y*z)/2u) {
        return;
    }

    let replica_index = index / ((t*x*y*z)/2u);
    let replica_base_offset = replica_index * (t*x*y*z*p);
    index = index % ((t*x*y*z)/2u);

    let vn_offset = i32(dim_indices.data[5u + replica_index]);

    let mu = dim_indices.data[5u + num_replicas + 0u];
    let nu = dim_indices.data[5u + num_replicas + 1u];
    let sigma = dim_indices.data[5u + num_replicas + 2u];
    let rho = dim_indices.data[5u + num_replicas + 3u];
    // Assertion: mu < nu < sigma
    // rho is just the leftover
    let offset = dim_indices.data[5u + num_replicas + 4u];

    let rho_index = index / (dim_indices.data[mu] * dim_indices.data[nu] * dim_indices.data[sigma] /2u);
    index = index % (dim_indices.data[mu] * dim_indices.data[nu] * dim_indices.data[sigma]/2u);
    let mu_index = index / (dim_indices.data[nu] * dim_indices.data[sigma]/2u);
    index = index % (dim_indices.data[nu] * dim_indices.data[sigma]/2u);
    let nu_index = index / (dim_indices.data[sigma]/2u);
    index = index % (dim_indices.data[sigma]/2u);
    let sigma_index_half = index;

    let parity = (mu_index + nu_index + offset) % 2u;
    let sigma_index = 2u*sigma_index_half + parity;


    var cube_index : vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
    cube_index[mu] = mu_index;
    cube_index[nu] = nu_index;
    cube_index[sigma] = sigma_index;
    cube_index[rho] = rho_index;


    var pos_indices : vec3<u32> = vec3<u32>(0u, 0u, 0u);
    var neg_indices : vec3<u32> = vec3<u32>(0u, 0u, 0u);

    // First do the ones attached to the calculated point
    let first = mu;
    let second = nu;
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    pos_indices[0] = index;

    let first = mu;
    let second = sigma;
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    neg_indices[0] = index;

    let first = nu;
    let second = sigma;
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    pos_indices[1] = index;

    // Now the opposing faces.
    let first = mu;
    let second = nu;
    let normal = sigma;
    cube_index[normal] = (cube_index[normal] + 1u) % dim_indices.data[normal];
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    cube_index[normal] = (cube_index[normal] + dim_indices.data[normal] - 1u) % dim_indices.data[normal];
    neg_indices[1] = index;

    let first = mu;
    let second = sigma;
    let normal = nu;
    cube_index[normal] = (cube_index[normal] + 1u) % dim_indices.data[normal];
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    cube_index[normal] = (cube_index[normal] + dim_indices.data[normal] - 1u) % dim_indices.data[normal];
    pos_indices[2] = index;

    let first = nu;
    let second = sigma;
    let normal = mu;
    cube_index[normal] = (cube_index[normal] + 1u) % dim_indices.data[normal];
    let sign = (second - first) % 2u;
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    cube_index[normal] = (cube_index[normal] + dim_indices.data[normal] - 1u) % dim_indices.data[normal];
    neg_indices[2] = index;

    // Now we have the positive and negative indices.
    var add_one_dv = f32(0.0);
    var sub_one_dv = f32(0.0);
    for(var i: i32 = 0; i < 3; i = i + 1) {
        let pos_index = pos_indices[i];
        let neg_index = neg_indices[i];
        let pos_np = state.state[replica_base_offset + pos_index];
        let neg_np = state.state[replica_base_offset + neg_index];
        // add one to pos_index, subtract one from neg_index
        add_one_dv = add_one_dv + vn.vn[vn_offset + abs(pos_np + 1)] + vn.vn[vn_offset + abs(neg_np - 1)] - vn.vn[vn_offset + abs(pos_np)] - vn.vn[vn_offset + abs(neg_np)];
        // subtract one to pos_index, add one from neg_index
        sub_one_dv = sub_one_dv + vn.vn[vn_offset + abs(pos_np - 1)] + vn.vn[vn_offset + abs(neg_np + 1)] - vn.vn[vn_offset + abs(pos_np)] - vn.vn[vn_offset + abs(neg_np)];
    }
    let add_one_p = exp(-add_one_dv);
    let sub_one_p = exp(-sub_one_dv);

    let random_float = prng(global_id.x);
    let random_float = random_float * (1.0 + add_one_p + sub_one_p) - 1.0;

    if (random_float < 0.0) {
        return;
    }

    // select(t,f,condition)
    let choice = select(-1, 1, random_float < add_one_p);
    let index = pos_indices[0];

    // Apply choice
    for(var i: i32 = 0; i < 3; i = i + 1) {
        let pos_index = pos_indices[i];
        let neg_index = neg_indices[i];
        let pos_np = state.state[replica_base_offset + pos_index];
        let neg_np = state.state[replica_base_offset + neg_index];
        state.state[replica_base_offset + pos_index] = pos_np + choice;
        state.state[replica_base_offset + neg_index] = neg_np - choice;
    }
}