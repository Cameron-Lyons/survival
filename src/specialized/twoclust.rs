#![allow(dead_code)]
pub fn has_multiple_clusters(id: &[i32], cluster: &[i32], idord: &[i32]) -> bool {
    let n = id.len();
    let mut i = 0;
    while i < n {
        let current_idx = idord[i] as usize;
        let current_id = id[current_idx];
        let expected_cluster = cluster[current_idx];
        while i < n {
            let idx = idord[i] as usize;
            if id[idx] != current_id {
                break;
            }
            if cluster[idx] != expected_cluster {
                return true;
            }
            i += 1;
        }
    }
    false
}
