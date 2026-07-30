pub(crate) mod cox_py_callback;
pub(crate) mod pyears3b;
pub(crate) mod pystep;

pub(crate) fn column_major_index(indices: &[usize], dims: &[usize]) -> usize {
    let mut index = 0;
    let mut stride = 1;
    for (&i, &dim) in indices.iter().zip(dims.iter()) {
        index += i * stride;
        stride *= dim;
    }
    index
}
