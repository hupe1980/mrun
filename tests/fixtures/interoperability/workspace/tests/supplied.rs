use mrun_public_read_edit_test::triangular_sum;

#[test]
fn includes_upper_bound_and_accepts_zero() {
    assert_eq!(triangular_sum(0), 0);
    assert_eq!(triangular_sum(1), 1);
    assert_eq!(triangular_sum(4), 10);
    assert_eq!(triangular_sum(100), 5050);
}