//! Smoke test: build two arrays, add them on the CPU stream, evaluate and read back.

use mlx_sys::*;

#[test]
fn add_on_cpu_stream() {
    let a_data = [1.0f32, 2.0, 3.0, 4.0];
    let b_data = [10.0f32, 20.0, 30.0, 40.0];
    let shape = [2i32, 2];
    // SAFETY: the data pointers and shape outlive the calls (mlx copies the data), every handle
    // created here is freed exactly once, and the result pointer is read only while `sum` lives.
    unsafe {
        let a = mlx_array_new_data(
            a_data.as_ptr().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            mlx_dtype__MLX_FLOAT32,
        );
        let b = mlx_array_new_data(
            b_data.as_ptr().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            mlx_dtype__MLX_FLOAT32,
        );
        let stream = mlx_default_cpu_stream_new();
        let mut sum = mlx_array_new();

        assert_eq!(mlx_add(&mut sum, a, b, stream), 0, "mlx_add failed");
        assert_eq!(mlx_array_eval(sum), 0, "mlx_array_eval failed");

        assert_eq!(mlx_array_ndim(sum), 2);
        assert_eq!(mlx_array_size(sum), 4);
        assert_eq!(mlx_array_dtype(sum), mlx_dtype__MLX_FLOAT32);
        let data = mlx_array_data_float32(sum);
        assert!(!data.is_null());
        let values = std::slice::from_raw_parts(data, 4);
        assert_eq!(values, &[11.0, 22.0, 33.0, 44.0]);

        assert_eq!(mlx_array_free(sum), 0);
        assert_eq!(mlx_array_free(b), 0);
        assert_eq!(mlx_array_free(a), 0);
        assert_eq!(mlx_stream_free(stream), 0);
    }
}
