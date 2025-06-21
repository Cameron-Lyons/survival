// use extendr_api::prelude::*;

// #[extendr]
// fn pystep(
//     nc: i32,
//     data: Doubles,
//     fac: Integers,
//     dims: Integers,
//     cuts: List,
//     step: f64,
//     edge: i32,
// ) -> List {
//     let nc = nc as usize;
//     let mut data_vec: Vec<f64> = data.iter().collect();
//     let fac_vec: Vec<i32> = fac.iter().collect();
//     let dims_vec: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
//     let cuts_slices: Vec<&[f64]> = cuts
//         .iter()
//         .map(|c| c.as_real_vector().unwrap().as_slice())
//         .collect();
//     let edge = edge != 0;
//
//     let mut kk = 1;
//     let mut index = 0;
//     let mut index2_initial = 0;
//     let mut wt = 1.0;
//     let mut shortfall = 0.0;
//     let mut maxtime = step;
//
//     for i in 0..nc {
//         if fac_vec[i] == 1 {
//             let category = data_vec[i] as usize - 1;
//             index += category * kk;
//             kk *= dims_vec[i];
//         } else {
//             let dtemp = if fac_vec[i] > 1 {
//                 1 + (fac_vec[i] - 1) as usize * dims_vec[i]
//             } else {
//                 dims_vec[i]
//             };
//             let cuts_i = cuts_slices[i];
//             let j = cuts_i.partition_point(|&x| x <= data_vec[i]);
//
//             let mut temp;
//             let mut j_adjusted = j;
//
//             if j == 0 {
//                 temp = cuts_i[0] - data_vec[i];
//                 if !edge {
//                     if temp > shortfall {
//                         shortfall = temp.min(step);
//                     }
//                 }
//                 maxtime = maxtime.min(temp);
//             } else if j == dtemp {
//                 if !edge {
//                     if let Some(last_cut) = cuts_i.last() {
//                         temp = *last_cut - data_vec[i];
//                         if temp <= 0.0 {
//                             shortfall = step;
//                         } else {
//                             maxtime = maxtime.min(temp);
//                         }
//                     } else {
//                         shortfall = step;
//                     }
//                 }
//                 j_adjusted = if fac_vec[i] > 1 {
//                     dims_vec[i] - 1
//                 } else {
//                     dtemp - 1
//                 };
//             } else {
//                 temp = cuts_i[j] - data_vec[i];
//                 maxtime = maxtime.min(temp);
//                 j_adjusted = j - 1;
//
//                 if fac_vec[i] > 1 {
//                     let granularity = fac_vec[i] as usize;
//                     let offset = j_adjusted % granularity;
//                     wt = 1.0 - (offset as f64) / (granularity as f64);
//                     j_adjusted /= granularity;
//                     index2_initial = kk;
//                 }
//             }
//
//             index += j_adjusted * kk;
//             kk *= dims_vec[i];
//         }
//     }
//
//     let index2_total = index + index2_initial;
//     let (time, idx1, idx2, wt_final) = if shortfall > 0.0 {
//         (shortfall, -1, 0, 0.0)
//     } else {
//         (maxtime, index as i32, index2_total as i32, wt)
//     };
//
//     list!(time = time, index = idx1, index2 = idx2, wt = wt_final)
// }

// extendr_module! {
//     mod pystep;
//     fn pystep;
// }
