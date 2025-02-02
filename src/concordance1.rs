use extendr_api::prelude::*;

#[extendr]
fn concordance1(y: &[f64], wt: &[f64], indx: &[i32], ntree: i32) -> Vec<f64> {
    let n = wt.len();
    let ntree = ntree as usize;
    let mut count = vec![0.0; 5];
    let mut twt = vec![0.0; 2 * ntree];
    let (time, status) = (&y[0..n], &y[n..2 * n]);
    let mut vss = 0.0;

    let mut i = n as i32 - 1;
    while i >= 0 {
        let mut ndeath = 0.0;
        let mut j = i;

        if status[i as usize] == 1.0 {
            while j >= 0 && status[j as usize] == 1.0 && time[j as usize] == time[i as usize] {
                let j_idx = j as usize;
                ndeath += wt[j_idx];
                let index = indx[j_idx] as usize;

                for k in (j + 1)..=i {
                    count[3] += wt[j_idx] * wt[k as usize];
                }

                count[2] += wt[j_idx] * twt[ntree + index];

                let mut child = 2 * index + 1;
                if child < ntree {
                    count[0] += wt[j_idx] * twt[child];
                }

                child += 1;
                if child < ntree {
                    count[1] += wt[j_idx] * twt[child];
                }

                let mut current = index;
                while current > 0 {
                    let parent = (current - 1) / 2;
                    if current % 2 == 1 {
                        count[1] += wt[j_idx] * (twt[parent] - twt[current]);
                    } else {
                        count[0] += wt[j_idx] * (twt[parent] - twt[current]);
                    }
                    current = parent;
                }

                j -= 1;
            }
        }

        for idx in (j + 1)..=i {
            let i_idx = idx as usize;
            let mut wsum1 = 0.0;
            let oldmean = twt[0] / 2.0;
            let index = indx[i_idx] as usize;

            twt[ntree + index] += wt[i_idx];
            twt[index] += wt[i_idx];
            let wsum2 = twt[ntree + index];

            let mut child = 2 * index + 1;
            if child < ntree {
                wsum1 += twt[child];
            }

            let mut current = index;
            while current > 0 {
                let parent = (current - 1) / 2;
                twt[parent] += wt[i_idx];
                if current % 2 == 0 {
                    wsum1 += twt[parent] - twt[current];
                }
                current = parent;
            }

            let wsum3 = twt[0] - (wsum1 + wsum2);
            let lmean = wsum1 / 2.0;
            let umean = wsum1 + wsum2 + wsum3 / 2.0;
            let newmean = twt[0] / 2.0;
            let myrank = wsum1 + wsum2 / 2.0;

            vss += wsum1 * (newmean + oldmean - 2.0 * lmean) * (newmean - oldmean);
            vss += wsum3 * (newmean + oldmean + wt[i_idx] - 2.0 * umean) * (oldmean - newmean);
            vss += wt[i_idx] * (myrank - newmean).powi(2);
        }

        if twt[0] > 0.0 {
            count[4] += ndeath * vss / twt[0];
        }

        i = j;
    }

    count
}

#[extendr]
fn concordance2(
    y: &[f64],
    wt: &[f64],
    indx: &[i32],
    ntree: i32,
    sortstop: &[i32],
    sortstart: &[i32],
) -> Vec<f64> {
    let n = wt.len();
    let ntree = ntree as usize;
    let mut count = vec![0.0; 5];
    let mut twt = vec![0.0; 2 * ntree];
    let (time1, time2, status) = (&y[0..n], &y[n..2 * n], &y[2 * n..3 * n]);
    let mut vss = 0.0;
    let mut istart = 0;

    let mut i = 0;
    while i < n {
        let iptr = sortstop[i] as usize;
        let mut ndeath = 0.0;
        let mut j = i;

        if status[iptr] == 1.0 {
            let dtime = time2[iptr];
            while istart < n && time1[sortstart[istart] as usize] >= dtime {
                let jptr = sortstart[istart] as usize;
                let mut wsum1 = 0.0;
                let oldmean = twt[0] / 2.0;
                let index = indx[jptr] as usize;

                twt[ntree + index] -= wt[jptr];
                twt[index] -= wt[jptr];
                let wsum2 = twt[ntree + index];

                let mut child = 2 * index + 1;
                if child < ntree {
                    wsum1 += twt[child];
                }

                let mut current = index;
                while current > 0 {
                    let parent = (current - 1) / 2;
                    twt[parent] -= wt[jptr];
                    if current % 2 == 0 {
                        wsum1 += twt[parent] - twt[current];
                    }
                    current = parent;
                }

                let wsum3 = twt[0] - (wsum1 + wsum2);
                let lmean = wsum1 / 2.0;
                let umean = wsum1 + wsum2 + wsum3 / 2.0;
                let newmean = twt[0] / 2.0;
                let myrank = wsum1 + wsum2 / 2.0;

                vss += wsum1 * (newmean + oldmean - 2.0 * lmean) * (newmean - oldmean);
                let oldmean = oldmean - wt[jptr];
                vss += wsum3 * (newmean + oldmean - 2.0 * umean) * (newmean - oldmean);
                vss -= wt[jptr] * (myrank - newmean).powi(2);

                istart += 1;
            }

            while j < n
                && status[sortstop[j] as usize] == 1.0
                && time2[sortstop[j] as usize] == dtime
            {
                let jptr = sortstop[j] as usize;
                ndeath += wt[jptr];
                let index = indx[jptr] as usize;

                for k in i..j {
                    count[3] += wt[jptr] * wt[sortstop[k] as usize];
                }

                count[2] += wt[jptr] * twt[ntree + index];

                let mut child = 2 * index + 1;
                if child < ntree {
                    count[0] += wt[jptr] * twt[child];
                }

                child += 1;
                if child < ntree {
                    count[1] += wt[jptr] * twt[child];
                }

                let mut current = index;
                while current > 0 {
                    let parent = (current - 1) / 2;
                    if current % 2 == 1 {
                        count[1] += wt[jptr] * (twt[parent] - twt[current]);
                    } else {
                        count[0] += wt[jptr] * (twt[parent] - twt[current]);
                    }
                    current = parent;
                }

                j += 1;
            }
        }

        for idx in i..j {
            let iptr = sortstop[idx] as usize;
            let mut wsum1 = 0.0;
            let oldmean = twt[0] / 2.0;
            let index = indx[iptr] as usize;

            twt[ntree + index] += wt[iptr];
            twt[index] += wt[iptr];
            let wsum2 = twt[ntree + index];

            let mut child = 2 * index + 1;
            if child < ntree {
                wsum1 += twt[child];
            }

            let mut current = index;
            while current > 0 {
                let parent = (current - 1) / 2;
                twt[parent] += wt[iptr];
                if current % 2 == 0 {
                    wsum1 += twt[parent] - twt[current];
                }
                current = parent;
            }

            let wsum3 = twt[0] - (wsum1 + wsum2);
            let lmean = wsum1 / 2.0;
            let umean = wsum1 + wsum2 + wsum3 / 2.0;
            let newmean = twt[0] / 2.0;
            let myrank = wsum1 + wsum2 / 2.0;

            vss += wsum1 * (newmean + oldmean - 2.0 * lmean) * (newmean - oldmean);
            vss += wsum3 * (newmean + oldmean + wt[iptr] - 2.0 * umean) * (oldmean - newmean);
            vss += wt[iptr] * (myrank - newmean).powi(2);
        }

        if twt[0] > 0.0 {
            count[4] += ndeath * vss / twt[0];
        }

        i = j;
    }

    count
}

extendr_module! {
    mod survivalutils;
    fn concordance1;
    fn concordance2;
}
