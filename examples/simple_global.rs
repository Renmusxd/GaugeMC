use gaugemc::NDDualGraph;

fn main() {
    let mut graph = NDDualGraph::new(4, 4, 4, 4, [0.0]);
    let mut choices = vec![0; graph.num_planes()];
    choices[0] = 1;
    graph.apply_global_updates(&choices);
}
