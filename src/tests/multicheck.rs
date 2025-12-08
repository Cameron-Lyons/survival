#![allow(dead_code)]
pub fn multicheck(
    time1: &[f64],
    time2: &[f64],
    status: &[i32],
    id: &[i32],
    istate: &[i32],
    sort: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let n = id.len();
    assert_eq!(time1.len(), n);
    assert_eq!(time2.len(), n);
    assert_eq!(status.len(), n);
    assert_eq!(istate.len(), n);
    assert_eq!(sort.len(), n);

    let mut dupid = vec![0; n];
    let mut gap = vec![0; n];
    let mut cstate = vec![0; n];

    let mut oldid = -1i32;
    let mut oldii = 0usize;

    for (i, &s) in sort.iter().enumerate() {
        let ii = s as usize;

        if id[ii] == oldid {
            dupid[ii] = 0;

            if time1[ii] == time2[oldii] {
                gap[ii] = 0;
            } else if time1[ii] > time2[oldii] {
                gap[ii] = 1;
            } else {
                gap[ii] = -1;
            }

            if status[oldii] > 0 {
                cstate[ii] = status[oldii];
            } else {
                cstate[ii] = cstate[oldii];
            }
        } else {
            oldid = id[ii];
            dupid[ii] = 0;
            gap[ii] = 0;
            cstate[ii] = istate[ii];

            if i > 0 {
                dupid[oldii] += 2;
            }
        }

        oldii = ii;
    }

    if n > 0 {
        dupid[oldii] += 2;
    }

    (dupid, gap, cstate)
}
