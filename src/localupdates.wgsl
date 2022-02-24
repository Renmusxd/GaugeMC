
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

[[group(0), binding(0)]]
var<storage, read_write> state : State;
[[group(0), binding(1)]]
var<storage, read> vn : Vn;
[[group(0), binding(2)]]
var<storage, read> dim_indices : DimInformation;


fn p_from_dims(first: u32, second: u32) -> u32 {
    // first and second are between 0x00 and 0x11
    // outputs from 0x000 to 0x101
    // This logic made by Karnaugh map with invalid entries replaced by useful values.
    let a = ((first == u32(0x01)) | (first == u32(0x10))) & (second == u32(0x11));
    let b = ((first == u32(0x00)) & (second == u32(0x11))) | ((first == u32(0x01)) & (second == u32(0x10)));
    let c = (first == u32(0x10)) | (second == u32(0x10));
    return (u32(a) << u32(2)) | (u32(b) << u32(1)) | u32(c);
}

[[stage(compute), workgroup_size(1,1,1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let mu = dim_indices.data[4 + 0];
    let nu = dim_indices.data[4 + 1];
    let sigma = dim_indices.data[4 + 2];
    let rho = dim_indices.data[4 + 3];

    // TODO May be an issue here.
    var index = global_id.x + u32(0);
    let rho_index = index / (dim_indices.data[mu] * dim_indices.data[nu] * dim_indices.data[sigma]);
    index = index % (dim_indices.data[mu] * dim_indices.data[nu] * dim_indices.data[sigma]);
    let mu_index = index / (dim_indices.data[nu] * dim_indices.data[sigma]);
    index = index % (dim_indices.data[nu] * dim_indices.data[sigma]);
    let nu_index = index / dim_indices.data[sigma];
    index = index % dim_indices.data[sigma];
    let sigma_index_half = index;

    let offset = dim_indices.data[8];
    let parity = (mu_index + nu_index + offset) % u32(2);
    let sigma_index = u32(2)*sigma_index_half + parity;

    var cube_index : vec4<u32> = vec4<u32>(u32(0), u32(0), u32(0), u32(0));
    cube_index[mu] = mu_index;
    cube_index[nu] = nu_index;
    cube_index[sigma] = sigma_index;
    cube_index[rho] = rho_index;

    let t = dim_indices.data[0];
    let x = dim_indices.data[1];
    let y = dim_indices.data[2];
    let z = dim_indices.data[3];
    let p = u32(6);

    var pos_indices : vec3<u32> = vec3<u32>(u32(0), u32(0), u32(0));
    var neg_indices : vec3<u32> = vec3<u32>(u32(0), u32(0), u32(0));
    var pos_count: u32 = u32(0);
    var neg_count: u32 = u32(0);

    // First do the ones attached to the calculated point
    let first = mu;
    let second = nu;
    let sign = (second - first + u32(1)) % u32(2);
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    if (sign == u32(0)) {
        pos_indices[pos_count] = index;
        pos_count = pos_count + u32(1);
    } else {
        neg_indices[neg_count] = index;
        neg_count = neg_count + u32(1);
    }

    let first = mu;
    let second = sigma;
    let sign = (second - first + u32(1)) % u32(2);
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    if (sign == u32(0)) {
        pos_indices[pos_count] = index;
        pos_count = pos_count + u32(1);
    } else {
        neg_indices[neg_count] = index;
        neg_count = neg_count + u32(1);
    }

    let first = nu;
    let second = sigma;
    let sign = (second - first + u32(1)) % u32(2);
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    if (sign == u32(0)) {
        pos_indices[pos_count] = index;
        pos_count = pos_count + u32(1);
    } else {
        neg_indices[neg_count] = index;
        neg_count = neg_count + u32(1);
    }

    // Now the opposing faces.
    let first = mu;
    let second = nu;
    let normal = sigma;
    cube_index[normal] = (cube_index[normal] + u32(1)) % dim_indices.data[normal];
    let sign = (second - first) % u32(2);
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    cube_index[normal] = (cube_index[normal] + dim_indices.data[normal] - u32(1)) % dim_indices.data[normal];
    if (sign == u32(0)) {
        pos_indices[pos_count] = index;
        pos_count = pos_count + u32(1);
    } else {
        neg_indices[neg_count] = index;
        neg_count = neg_count + u32(1);
    }

    let first = mu;
    let second = sigma;
    let normal = nu;
    cube_index[normal] = (cube_index[normal] + u32(1)) % dim_indices.data[normal];
    let sign = (second - first) % u32(2);
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    cube_index[normal] = (cube_index[normal] + dim_indices.data[normal] - u32(1)) % dim_indices.data[normal];
    if (sign == u32(0)) {
        pos_indices[pos_count] = index;
        pos_count = pos_count + u32(1);
    } else {
        neg_indices[neg_count] = index;
        neg_count = neg_count + u32(1);
    }

    let first = nu;
    let second = sigma;
    let normal = mu;
    cube_index[normal] = (cube_index[normal] + u32(1)) % dim_indices.data[normal];
    let sign = (second - first) % u32(2);
    let p_index = p_from_dims(first, second);
    let index = cube_index[0]*(x*y*z*p) + cube_index[1]*(y*z*p) + cube_index[2]*(z*p) + cube_index[3]*p + p_index;
    cube_index[normal] = (cube_index[normal] + dim_indices.data[normal] - u32(1)) % dim_indices.data[normal];
    if (sign == u32(0)) {
        pos_indices[pos_count] = index;
        pos_count = pos_count + u32(1);
    } else {
        neg_indices[neg_count] = index;
        neg_count = neg_count + u32(1);
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

    // TODO Make choice
    let choice = 1;

    // Apply choice
    for(var i: i32 = 0; i < 3; i = i + 1) {
        let pos_index = pos_indices[i];
        let neg_index = neg_indices[i];
        let pos_np = state.state[pos_index];
        let neg_np = state.state[neg_index];
        state.state[pos_index] = pos_np + choice;
        state.state[neg_index] = neg_np - choice;
    }
}