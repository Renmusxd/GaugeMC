
struct State {
  state : array<i32>;
};

struct Vn {
    vn : array<f32>;
};

struct DimInformation {
    // First the dimensions of the graph
    // t, x, y, z
    // Then the cube selections (first 3 are 3D cube, last is remaining)
    // mu, nu, sigma, rho, offset
    data : array<u32>;
};

struct PCGState {
  state : array<u32>;
};

[[group(0), binding(0)]]
var<storage, read_write> state : State;
[[group(0), binding(1)]]
var<storage, read> vn : Vn;
[[group(0), binding(2)]]
var<storage, read> dim_indices : DimInformation;
[[group(0), binding(3)]]
var<storage, read_write> pcgstate : PCGState;

fn p_from_dims(first: u32, second: u32) -> u32 {
    // first and second are between 0x00 and 0x11
    // outputs from 0x000 to 0x101
    // This logic made by Karnaugh map with invalid entries replaced by useful values.
    let a = ((first == 1u) | (first == 2u)) & (second == 3u);
    let b = ((first == 0u) & (second == 3u)) | ((first == 1u) & (second == 2u));
    let c = (first == 2u) | (second == 2u);
    return (u32(a) << 2u) | (u32(b) << 1u) | u32(c);
}

// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
fn pcg_hash(input: u32) -> u32 {
    let state : u32 = input * 747796405u + 2891336453u;
    let word : u32 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

[[stage(compute), workgroup_size(256,1,1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    var index = global_id.x;

    let t = dim_indices.data[0];
    let x = dim_indices.data[1];
    let y = dim_indices.data[2];
    let z = dim_indices.data[3];
    let p = 6u;

    if (global_id.x >= (t*x*y*z)/2u) {
        return;
    }

    let mu = dim_indices.data[4 + 0];
    let nu = dim_indices.data[4 + 1];
    let sigma = dim_indices.data[4 + 2];
    let rho = dim_indices.data[4 + 3];

    let rho_index = index / (dim_indices.data[mu] * dim_indices.data[nu] * dim_indices.data[sigma] /2u);
    index = index % (dim_indices.data[mu] * dim_indices.data[nu] * dim_indices.data[sigma]/2u);
    let mu_index = index / (dim_indices.data[nu] * dim_indices.data[sigma]/2u);
    index = index % (dim_indices.data[nu] * dim_indices.data[sigma]/2u);
    let nu_index = index / (dim_indices.data[sigma]/2u);
    index = index % (dim_indices.data[sigma]/2u);
    let sigma_index_half = index;

    let offset = dim_indices.data[8];
    let parity = (mu_index + nu_index + offset) % 2u;
    let sigma_index = 2u*sigma_index_half + parity;

    var cube_index : vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
    cube_index[mu] = mu_index;
    cube_index[nu] = nu_index;
    cube_index[sigma] = sigma_index;
    cube_index[rho] = rho_index;


    var pos_indices : vec3<u32> = vec3<u32>(0u, 0u, 0u);
    var neg_indices : vec3<u32> = vec3<u32>(0u, 0u, 0u);
    var pos_count: u32 = 0u;
    var neg_count: u32 = 0u;

    // First do the ones attached to the calculated point
    let first = mu;
    let second = nu;
    let sign = (second - first + 1u) % 2u;
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    if (sign == 0u) {
        pos_indices[pos_count] = index;
        pos_count = pos_count + 1u;
    } else {
        neg_indices[neg_count] = index;
        neg_count = neg_count + 1u;
    }

    let first = mu;
    let second = sigma;
    let sign = (second - first + 1u) % 2u;
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    if (sign == 0u) {
        pos_indices[pos_count] = index;
        pos_count = pos_count + 1u;
    } else {
        neg_indices[neg_count] = index;
        neg_count = neg_count + 1u;
    }

    let first = nu;
    let second = sigma;
    let sign = (second - first + 1u) % 2u;
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    if (sign == 0u) {
        pos_indices[pos_count] = index;
        pos_count = pos_count + 1u;
    } else {
        neg_indices[neg_count] = index;
        neg_count = neg_count + 1u;
    }

    // Now the opposing faces.
    let first = mu;
    let second = nu;
    let normal = sigma;
    cube_index[normal] = (cube_index[normal] + 1u) % dim_indices.data[normal];
    let sign = (second - first) % 2u;
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    cube_index[normal] = (cube_index[normal] + dim_indices.data[normal] - 1u) % dim_indices.data[normal];
    if (sign == 0u) {
        pos_indices[pos_count] = index;
        pos_count = pos_count + 1u;
    } else {
        neg_indices[neg_count] = index;
        neg_count = neg_count + 1u;
    }

    let first = mu;
    let second = sigma;
    let normal = nu;
    cube_index[normal] = (cube_index[normal] + 1u) % dim_indices.data[normal];
    let sign = (second - first) % 2u;
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    cube_index[normal] = (cube_index[normal] + dim_indices.data[normal] - 1u) % dim_indices.data[normal];
    if (sign == 0u) {
        pos_indices[pos_count] = index;
        pos_count = pos_count + 1u;
    } else {
        neg_indices[neg_count] = index;
        neg_count = neg_count + 1u;
    }

    let first = nu;
    let second = sigma;
    let normal = mu;
    cube_index[normal] = (cube_index[normal] + 1u) % dim_indices.data[normal];
    let sign = (second - first) % 2u;
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    cube_index[normal] = (cube_index[normal] + dim_indices.data[normal] - 1u) % dim_indices.data[normal];
    if (sign == 0u) {
        pos_indices[pos_count] = index;
        pos_count = pos_count + 1u;
    } else {
        neg_indices[neg_count] = index;
        neg_count = neg_count + 1u;
    }

    // Now we have the positive and negative indices.
    var add_one_dv = f32(0.0);
    var sub_one_dv = f32(0.0);
    for(var i: i32 = 0; i < 3; i = i + 1) {
        let pos_index = pos_indices[i];
        let neg_index = neg_indices[i];
        let pos_np = state.state[pos_index];
        let neg_np = state.state[neg_index];
        // add one to pos_index, subtract one from neg_index
        add_one_dv = add_one_dv + vn.vn[abs(pos_np + 1)] + vn.vn[abs(neg_np - 1)] - vn.vn[abs(pos_np)] - vn.vn[abs(neg_np)];
        // subtract one to pos_index, add one from neg_index
        sub_one_dv = sub_one_dv + vn.vn[abs(pos_np - 1)] + vn.vn[abs(neg_np + 1)] - vn.vn[abs(pos_np)] - vn.vn[abs(neg_np)];
    }
    let add_one_p = exp(-add_one_dv);
    let sub_one_p = exp(-sub_one_dv);

    pcgstate.state[global_id.x] = pcg_hash(pcgstate.state[global_id.x]);
    let random_float: f32 = f32(i32(pcgstate.state[global_id.x])) * 2.32830643653869628906e-010 + 0.5;
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
        let pos_np = state.state[pos_index];
        let neg_np = state.state[neg_index];
        state.state[pos_index] = pos_np + choice;
        state.state[neg_index] = neg_np - choice;
        // For debugging.
        //state.state[pos_index] = i32(global_id.x);
        //state.state[neg_index] = -i32(global_id.x);
    }
}