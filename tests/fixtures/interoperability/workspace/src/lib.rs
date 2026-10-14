pub fn triangular_sum(upper: u32) -> u64 {
    (1..upper).map(u64::from).sum()
}