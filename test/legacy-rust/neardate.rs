use chrono::NaiveDate;
use std::collections::HashMap;

fn neardate(
    df1_ids: &[i32],
    df1_dates: &[NaiveDate],
    df2_entries: &[(i32, NaiveDate)],
    best: &str,
) -> Vec<Option<usize>> {
    let mut df2_map: HashMap<i32, Vec<(usize, NaiveDate)>> = HashMap::new();
    for (original_index, (id, y2)) in df2_entries.iter().enumerate() {
        df2_map.entry(*id).or_default().push((original_index, *y2));
    }

    let mut result = Vec::with_capacity(df1_ids.len());
    for (current_id, current_date) in df1_ids.iter().zip(df1_dates) {
        let entries = df2_map.get(current_id);
        let res = entries.and_then(|entries| {
            let filtered: Vec<_> = entries
                .iter()
                .filter(|(_, y2)| match best {
                    "after" => y2 >= current_date,
                    "prior" => y2 <= current_date,
                    _ => panic!("Invalid 'best' parameter"),
                })
                .collect();

            if filtered.is_empty() {
                None
            } else {
                let target_date = match best {
                    "after" => filtered.iter().map(|(_, y2)| y2).min().unwrap(),
                    "prior" => filtered.iter().map(|(_, y2)| y2).max().unwrap(),
                    _ => unreachable!(),
                };

                let min_entry = filtered
                    .iter()
                    .filter(|(_, y2)| y2 == target_date)
                    .min_by_key(|(oi, _)| oi)
                    .unwrap();

                Some(min_entry.0)
            }
        });
        result.push(res);
    }
    result
}

fn main() {
    let df1_ids: Vec<i32> = (1..=10).collect();
    let df1_dates: Vec<NaiveDate> = [
        "1992-01-01",
        "1996-01-01",
        "1997-03-20",
        "2000-01-01",
        "2001-01-01",
        "2004-01-01",
        "2014-03-27",
        "2014-01-30",
        "2000-08-01",
        "1997-04-29",
    ]
    .iter()
    .map(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap())
    .collect();

    let df2_entries: Vec<(i32, NaiveDate)> = [
        (1, "1998-04-30"),
        (1, "2004-07-01"),
        (2, "1999-04-14"),
        (3, "2001-02-22"),
        (4, "2003-11-19"),
        (4, "2005-02-15"),
        (5, "2006-06-22"),
        (6, "2007-09-20"),
        (7, "2013-08-02"),
        (7, "2015-01-09"),
        (8, "2014-01-15"),
        (9, "2006-12-06"),
        (9, "1999-10-20"),
        (9, "2010-06-30"),
        (10, "1997-04-28"),
        (3, "1995-04-20"),
        (3, "1997-03-20"),
        (6, "1998-04-30"),
        (6, "1995-04-20"),
        (8, "2006-12-06"),
    ]
    .iter()
    .map(|(id, s)| (*id, NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()))
    .collect();

    let i1 = neardate(&df1_ids, &df1_dates, &df2_entries, "after");
    let expected_i1 = vec![
        Some(0),
        Some(2),
        Some(16),
        Some(4),
        Some(6),
        Some(7),
        Some(9),
        None,
        Some(11),
        None,
    ];
    assert_eq!(i1, expected_i1);

    let i2 = neardate(&df1_ids, &df1_dates, &df2_entries, "prior");
    let expected_i2 = vec![
        None,
        None,
        Some(16),
        None,
        None,
        Some(17),
        Some(8),
        Some(10),
        Some(13),
        Some(14),
    ];
    assert_eq!(i2, expected_i2);

    println!("All tests passed!");
}
