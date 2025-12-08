pub fn surv_concordance(
    n: usize,
    time: &[f64],
    status: &[i32],
    x: &[f64],
    n2: usize,
    x2: &[f64],
    temp: &mut [i32],
    result: &mut [i32],
) {
    for r in result.iter_mut() {
        *r = 0;
    }

    let (count1, count2) = temp.split_at_mut(n2);

    for c in count1.iter_mut() {
        *c = 0;
    }

    let mut tdeath = 0;

    for i in 0..n {
        if status[i] > 0 {
            let count = if tdeath == 0 { count1 } else { count2 };

            let mut nright = 0;
            let mut start = 0;
            let mut end = n2 - 1;
            let mut k = 0;

            while start <= end {
                k = (start + end) / 2;
                if x[i] == x2[k] {
                    break;
                } else if x[i] < x2[k] {
                    end = k - 1;
                    let right_start = k + 1;
                    let right_end = end;
                    if right_start <= right_end {
                        let mid = (right_start + right_end) / 2;
                        nright += count[mid];
                    }
                } else {
                    start = k + 1;
                }
            }

            let mut nsame = count[k];

            if k < end {
                let right_start = k + 1;
                let right_end = end;
                let mid = (right_start + right_end) / 2;
                let j = count[mid];
                nsame -= j;
                nright += j;
            }

            if k > start {
                let left_start = start;
                let left_end = k - 1;
                let mid = (left_start + left_end) / 2;
                let j = count[mid];
                nsame -= j;
            }

            result[3] += nsame;
            result[1] += nright;
            result[0] += (i - tdeath) - (nsame + nright);

            if i < n - 1 && status[i + 1] > 0 && time[i] == time[i + 1] {
                tdeath += 1;
                if tdeath == 1 {
                    count2.copy_from_slice(count1);
                }
            } else {
                result[2] += (tdeath * (tdeath + 1)) / 2;
                tdeath = 0;
            }
        } else {
            tdeath = 0;
            result[4] += i;
        }

        let mut start = 0;
        let mut end = n2 - 1;
        while start <= end {
            let k = (start + end) / 2;
            count1[k] += 1;
            if x[i] == x2[k] {
                break;
            } else if x[i] < x2[k] {
                end = k - 1;
            } else {
                start = k + 1;
            }
        }
    }
}
