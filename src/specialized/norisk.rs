#![allow(dead_code)]
pub fn norisk(
    time1: &[f64],
    time2: &[f64],
    status: &[i32],
    sort1: &[i32],
    sort2: &[i32],
    strata: &[i32],
) -> Vec<i32> {
    let n = time1.len();
    assert_eq!(time2.len(), n);
    assert_eq!(status.len(), n);
    assert_eq!(sort1.len(), n);
    assert_eq!(sort2.len(), n);
    assert!(strata.iter().all(|&s| s >= 0 && s <= n as i32));

    let mut notused = vec![0; n];
    let mut ndeath = 0;
    let mut istrat = 0;
    let mut j = 0;

    for i in 0..n {
        let p2 = sort2[i] as usize;
        let dtime = time2[p2];

        if i == strata.get(istrat).copied().unwrap_or(n as i32) as usize {
            while j < i {
                let p1 = sort1[j] as usize;
                notused[p1] = if ndeath > notused[p1] { 1 } else { 0 };
                j += 1;
            }
            ndeath = 0;
            istrat += 1;
        } else {
            while j < i && time1[sort1[j] as usize] >= dtime {
                let p1 = sort1[j] as usize;
                notused[p1] = if ndeath > notused[p1] { 1 } else { 0 };
                j += 1;
            }
        }

        ndeath += status[p2];
        if j < n {
            let p1 = sort1[j] as usize;
            notused[p1] = ndeath;
        }
    }

    while j < n {
        let p1 = sort1[j] as usize;
        notused[p1] = if ndeath > notused[p1] { 1 } else { 0 };
        j += 1;
    }

    notused
}
