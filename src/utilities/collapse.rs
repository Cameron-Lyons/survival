// TODO: This function requires extendr/R types (Robj, list!) which are not currently available
// Uncomment and add extendr dependency if needed
/*
fn collapse(y: &[f64], x: &[i32], istate: &[i32], id: &[i32], wt: &[f64], order: &[i32]) -> Robj {
    let n = id.len();
    assert_eq!(y.len(), 3 * n, "y must have 3 columns");
    assert_eq!(x.len(), n, "x length mismatch");
    assert_eq!(istate.len(), n, "istate length mismatch");
    assert_eq!(wt.len(), n, "wt length mismatch");
    assert_eq!(order.len(), n, "order length mismatch");

    let time1 = &y[0..n];
    let time2 = &y[n..2 * n];
    let status = &y[2 * n..3 * n];

    let mut i1 = Vec::new();
    let mut i2 = Vec::new();

    let mut i = 0;
    while i < n {
        let start_pos = i;
        let mut k1 = order[start_pos] as usize;

        let mut k = i + 1;
        while k < n {
            let k2 = order[k] as usize;
            if status[k1] != 0.0
                || id[k1] != id[k2]
                || x[k1] != x[k2]
                || (time1[k1] - time2[k2]).abs() > 1e-9
                || istate[k1] != istate[k2]
                || (wt[k1] - wt[k2]).abs() > 1e-9
            {
                break;
            }
            k1 = k2;
            i += 1;
            k += 1;
        }

        i1.push((k1 + 1) as i32);
        i2.push((order[start_pos] as usize + 1) as i32);
        i += 1;
    }

    let mut out_data = Vec::with_capacity(i1.len() * 2);
    out_data.extend(i2);
    out_data.extend(i1);

    Robj::new_matrix(i1.len(), 2, out_data)
        .expect("Failed to create matrix")
        .set_attrib("dimnames", list!(vec!["start", "end"]))
        .unwrap()
}
*/
